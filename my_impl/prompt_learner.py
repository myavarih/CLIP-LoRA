import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class PromptLearner(nn.Module):
    """
    CoOp-style Prompt Learner with optional Learnable Class Descriptors.
    
    Generates text prompts: [SOS, V_1, V_2, ..., V_M, CLASS_TOKENS, EOS, PADDING...]
    where V_1...V_M are M learnable context vectors.
    
    If csc is True:
      - Class-Specific Context (CSC): Each class has its own independent context vectors V_1^{(c)}...V_M^{(c)}.
    If learn_class_tokens is True:
      - 'residual': class_tokens = base_token_embedding + delta (delta initialized to 0)
      - 'full': class_tokens are fully trainable parameters initialized from class token embeddings.
    """
    def __init__(self, 
                 clip_model, 
                 classnames, 
                 n_ctx=4, 
                 ctx_init="", 
                 csc=False,
                 class_token_position='end',
                 learn_class_tokens=False,
                 class_token_mode='residual'):
        super().__init__()
        self.classnames = classnames
        self.num_classes = len(classnames)
        self.n_ctx = n_ctx
        self.csc = csc
        self.class_token_position = class_token_position
        self.learn_class_tokens = learn_class_tokens
        self.class_token_mode = class_token_mode
        
        dtype = clip_model.dtype
        ctx_dim = clip_model.transformer.width
        self.ctx_dim = ctx_dim
        self.dtype = dtype

        # 1. Initialize Context Vectors (V_1...V_M)
        embed_device = clip_model.token_embedding.weight.device
        
        if ctx_init and "{}" in ctx_init.replace("_", " "):
            ctx_init_clean = ctx_init.replace("_", " ")
            parts = ctx_init_clean.split("{}")
            prefix_str = parts[0].strip()
            suffix_str = parts[1].strip() if len(parts) > 1 else ""
            
            # Tokenize prefix and suffix separately
            tok_pref = clip.tokenize(prefix_str).to(embed_device)
            with torch.no_grad():
                emb_pref = clip_model.token_embedding(tok_pref).type(dtype)
            eos_pref = tok_pref.argmax().item()
            n_pref = eos_pref - 1
            pref_vectors = emb_pref[0, 1 : 1 + n_pref, :]
            
            if suffix_str:
                tok_suff = clip.tokenize(suffix_str).to(embed_device)
                with torch.no_grad():
                    emb_suff = clip_model.token_embedding(tok_suff).type(dtype)
                eos_suff = tok_suff.argmax().item()
                n_suff = eos_suff - 1
                suff_vectors = emb_suff[0, 1 : 1 + n_suff, :]
            else:
                n_suff = 0
                suff_vectors = torch.empty(0, ctx_dim, dtype=dtype, device=embed_device)
                
            self.prefix_len = n_pref
            self.suffix_len = n_suff
            self.n_ctx = n_pref + n_suff
            n_ctx = self.n_ctx
            
            ctx_vectors = torch.cat([pref_vectors, suff_vectors], dim=0) # [n_ctx, ctx_dim]
            prompt_prefix = ctx_init_clean
            if csc:
                ctx_vectors = ctx_vectors.unsqueeze(0).repeat(self.num_classes, 1, 1)
        elif ctx_init:
            ctx_init_clean = ctx_init.replace("_", " ")
            prompt_tokens = clip.tokenize(ctx_init_clean).to(embed_device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt_tokens).type(dtype)
            eos_idx = prompt_tokens.argmax().item()
            num_init_tokens = eos_idx - 1
            if num_init_tokens > 0 and num_init_tokens != n_ctx:
                n_ctx = num_init_tokens
                self.n_ctx = n_ctx
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :] # [n_ctx, ctx_dim]
            prompt_prefix = ctx_init_clean
            
            if class_token_position == 'end':
                self.prefix_len = n_ctx
                self.suffix_len = 0
            elif class_token_position == 'front':
                self.prefix_len = 0
                self.suffix_len = n_ctx
            elif class_token_position == 'middle':
                self.prefix_len = n_ctx // 2
                self.suffix_len = n_ctx - self.prefix_len
            else:
                self.prefix_len = n_ctx
                self.suffix_len = 0
                
            if csc:
                ctx_vectors = ctx_vectors.unsqueeze(0).repeat(self.num_classes, 1, 1)
        else:
            if class_token_position == 'end':
                self.prefix_len = n_ctx
                self.suffix_len = 0
            elif class_token_position == 'front':
                self.prefix_len = 0
                self.suffix_len = n_ctx
            elif class_token_position == 'middle':
                self.prefix_len = n_ctx // 2
                self.suffix_len = n_ctx - self.prefix_len
            else:
                self.prefix_len = n_ctx
                self.suffix_len = 0

            if csc:
                ctx_vectors = torch.empty(self.num_classes, n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)

        self.prompt_prefix = prompt_prefix
        self.ctx = nn.Parameter(ctx_vectors)  # [num_classes, n_ctx, ctx_dim] if csc else [n_ctx, ctx_dim]

        # 2. Extract Base Class Token Embeddings & Meta
        # Tokenize each class name to inspect length and base embeddings
        self.base_class_embeds = []
        self.class_lengths = []
        self.sos_embeds = []
        self.eos_embeds = []
        self.pad_embeds = []
        self.tokenized_prompts_list = []
        self.eos_indices = []

        with torch.no_grad():
            for c, name in enumerate(classnames):
                clean_name = name.replace("_", " ")
                tokenized = clip.tokenize(clean_name)[0].to(embed_device)  # [77]
                self.tokenized_prompts_list.append(tokenized)
                
                # In standard clip tokenizer:
                # tokenized[0] = SOS (49406)
                # tokenized[1 : eos_idx] = class name tokens
                # tokenized[eos_idx] = EOS (49407)
                eos_idx = tokenized.argmax().item()
                num_class_tokens = eos_idx - 1
                self.class_lengths.append(num_class_tokens)

                embed = clip_model.token_embedding(tokenized.unsqueeze(0)).type(dtype)[0] # [77, ctx_dim]
                
                sos_emb = embed[0:1] # [1, ctx_dim]
                cls_emb = embed[1 : 1 + num_class_tokens] # [L_c, ctx_dim]
                eos_emb = embed[eos_idx : eos_idx + 1] # [1, ctx_dim]
                
                # Padding tokens from eos_pos + 1 to 77
                num_pad = 77 - (1 + self.prefix_len + num_class_tokens + self.suffix_len + 1)
                if num_pad < 0:
                    raise ValueError(f"Prompt length exceeds context window 77 for class: {name}")
                
                pad_emb = embed[eos_idx + 1 : eos_idx + 1 + num_pad] # [num_pad, ctx_dim]

                self.sos_embeds.append(sos_emb)
                self.base_class_embeds.append(cls_emb)
                self.eos_embeds.append(eos_emb)
                self.pad_embeds.append(pad_emb)
                self.eos_indices.append(1 + self.prefix_len + num_class_tokens + self.suffix_len)

        # Register fixed parts as buffers (moved with .to(device))
        self.register_buffer('eos_indices_tensor', torch.tensor(self.eos_indices, dtype=torch.long))

        # 3. Learnable Class Tokens (if enabled)
        self.class_deltas = None
        self.class_embeds = None
        if self.learn_class_tokens:
            if self.class_token_mode == 'residual':
                self.class_deltas = nn.ParameterList([
                    nn.Parameter(torch.zeros(L_c, ctx_dim, dtype=dtype)) 
                    for L_c in self.class_lengths
                ])
            else: # 'full'
                self.class_embeds = nn.ParameterList([
                    nn.Parameter(cls_emb.clone()) 
                    for cls_emb in self.base_class_embeds
                ])

    def forward(self):
        """
        Constructs and returns prompt embeddings for all classes: [num_classes, 77, ctx_dim]
        """
        prompts = []

        for c in range(self.num_classes):
            ctx_c = self.ctx[c] if self.csc else self.ctx  # [n_ctx, ctx_dim]
            sos = self.sos_embeds[c].to(ctx_c.device)
            eos = self.eos_embeds[c].to(ctx_c.device)
            pad = self.pad_embeds[c].to(ctx_c.device)
            
            # Determine class token embedding
            if self.learn_class_tokens:
                if self.class_token_mode == 'residual':
                    cls_emb = self.base_class_embeds[c].to(ctx_c.device) + self.class_deltas[c]
                else:
                    cls_emb = self.class_embeds[c].to(ctx_c.device)
            else:
                cls_emb = self.base_class_embeds[c].to(ctx_c.device)

            if self.prefix_len > 0 and self.suffix_len > 0:
                ctx_1 = ctx_c[:self.prefix_len]
                ctx_2 = ctx_c[self.prefix_len:]
                prompt_c = torch.cat([sos, ctx_1, cls_emb, ctx_2, eos, pad], dim=0)
            elif self.prefix_len > 0:
                prompt_c = torch.cat([sos, ctx_c, cls_emb, eos, pad], dim=0)
            elif self.suffix_len > 0:
                prompt_c = torch.cat([sos, cls_emb, ctx_c, eos, pad], dim=0)
            else:
                prompt_c = torch.cat([sos, ctx_c, cls_emb, eos, pad], dim=0)

            prompts.append(prompt_c)

        prompts = torch.stack(prompts, dim=0) # [num_classes, 77, ctx_dim]
        return prompts


class CustomCoOpCLIP(nn.Module):
    """
    Unified Hybrid Model:
    - Text Encoder: Driven by PromptLearner / ResidualTemplatePromptLearner + CLIP Text Transformer (optionally with LoRA).
    - Vision Encoder: CLIP Vision Transformer (with LoRA applied).
    """
    def __init__(self, clip_model, prompt_learner):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner = prompt_learner
        self.logit_scale = clip_model.logit_scale

    @property
    def dtype(self):
        return self.clip_model.dtype

    def encode_text(self, prompts=None):
        if prompts is None:
            prompts = self.prompt_learner() # [K, 77, ctx_dim]
            
        x = prompts + self.clip_model.positional_embedding.type(self.dtype).to(prompts.device)
        x = x.permute(1, 0, 2)  # [77, K, ctx_dim] (LND)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # [K, 77, ctx_dim] (NLD)
        x = self.clip_model.ln_final(x).type(self.dtype)

        # Extract EOS token embeddings
        eos_indices = self.prompt_learner.eos_indices_tensor.to(x.device)
        proj = self.clip_model.text_projection.to(x.device).type(self.dtype)
        x = x[torch.arange(x.shape[0], device=x.device), eos_indices] @ proj
        text_features = x / x.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image(self, images):
        img_features = self.clip_model.encode_image(images.type(self.dtype))
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        return img_features

    def forward(self, images):
        text_features = self.encode_text()
        image_features = self.encode_image(images)
        scale = self.logit_scale.exp()
        logits = scale * (image_features @ text_features.t())
        return logits, image_features, text_features


class ResidualTemplatePromptLearner(nn.Module):
    """
    Residual-Template Prompt Learner (RT-LoRA):
    - Retains full natural language template: "a photo of a {} walnut, a type of walnut."
    - Class tokens have learnable residual: w_c + Delta w_c (Aggressive LR, e.g. 1e-3)
    - Template tokens have learnable residual: t_k + Delta t_k (Conservative LR, e.g. 1e-4)
    """
    def __init__(self, 
                 clip_model, 
                 classnames, 
                 template=None, 
                 learn_template_tokens=True, 
                 learn_class_tokens=True):
        super().__init__()
        self.classnames = classnames
        self.num_classes = len(classnames)
        dtype = clip_model.dtype
        ctx_dim = clip_model.transformer.width
        self.ctx_dim = ctx_dim
        self.dtype = dtype
        self.learn_template_tokens = learn_template_tokens
        self.learn_class_tokens = learn_class_tokens
        
        if template is None:
            template = "a photo of a {} walnut, a type of walnut."
        self.template_str = template
        
        embed_device = clip_model.token_embedding.weight.device
        
        self.base_embeds = []
        self.class_spans = [] # (start, end)
        self.class_lengths = []
        self.eos_indices = []
        
        with torch.no_grad():
            for c, name in enumerate(classnames):
                clean_name = name.replace("_", " ")
                full_text = template.format(clean_name)
                tok_full = clip.tokenize(full_text)[0].to(embed_device) # [77]
                tok_cls = clip.tokenize(clean_name)[0].to(embed_device)
                
                n_cls = tok_cls.argmax().item() - 1
                self.class_lengths.append(n_cls)
                cls_tokens = tok_cls[1:1+n_cls].tolist()
                full_list = tok_full.tolist()
                
                start = -1
                for i in range(len(full_list) - len(cls_tokens) + 1):
                    if full_list[i:i+len(cls_tokens)] == cls_tokens:
                        start = i
                        break
                if start == -1:
                    start = 5
                
                self.class_spans.append((start, start + n_cls))
                eos_idx = tok_full.argmax().item()
                self.eos_indices.append(eos_idx)
                
                embed = clip_model.token_embedding(tok_full.unsqueeze(0)).type(dtype)[0] # [77, ctx_dim]
                self.base_embeds.append(embed)
                
        self.register_buffer('base_embeds_tensor', torch.stack(self.base_embeds, dim=0)) # [K, 77, ctx_dim]
        self.register_buffer('eos_indices_tensor', torch.tensor(self.eos_indices, dtype=torch.long))
        
        # 1. Class-Specific Residuals (Full / High LR)
        self.class_deltas = None
        if self.learn_class_tokens:
            self.class_deltas = nn.ParameterList([
                nn.Parameter(torch.zeros(L_c, ctx_dim, dtype=dtype))
                for L_c in self.class_lengths
            ])
            
        # 2. Template-Token Residuals (Low / Conservative LR)
        self.template_deltas = None
        if self.learn_template_tokens:
            # Per-token shift across the 77 token positions (shared across classes)
            self.template_deltas = nn.Parameter(torch.zeros(77, ctx_dim, dtype=dtype))
            
    def forward(self):
        prompts = []
        base = self.base_embeds_tensor # [K, 77, ctx_dim]
        device = base.device
        dtype = base.dtype
        
        for c in range(self.num_classes):
            p = base[c].clone()
            eos = self.eos_indices_tensor[c].item()
            
            # Apply template delta to non-padding positions (up to eos_idx)
            if self.learn_template_tokens and self.template_deltas is not None:
                delta_t = self.template_deltas[:eos+1].to(device=device, dtype=dtype)
                p[:eos+1] = p[:eos+1] + delta_t
                
            # Apply class-specific residual
            if self.learn_class_tokens and self.class_deltas is not None:
                start, end = self.class_spans[c]
                delta_c = self.class_deltas[c].to(device=device, dtype=dtype)
                p[start:end] = p[start:end] + delta_c
                
            prompts.append(p)
            
        return torch.stack(prompts, dim=0) # [num_classes, 77, ctx_dim]

