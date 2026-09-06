#!/usr/bin/env python3
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

OUT = os.path.join(os.path.dirname(__file__), 'figures2')
os.makedirs(OUT, exist_ok=True)

font_bold_path = fm.findfont('DejaVu Sans:weight=bold')
font_path = fm.findfont('DejaVu Sans')

def get_plain_lora_path(shot, mode):
    if mode == 'tsne':
        if shot == 1:
            return 'docs/figures/a1.5d0.4_final_test_tsne_1shots.png'
        elif shot == 4:
            return 'docs/figures/4shot_baseline_tsne.png'
        elif shot == 16:
            return 'docs/figures/s1_16shot_baseline_tsne.png'
        else:
            return 'docs/figures/32shot_baseline_tsne.png'
    else: # pca
        if shot == 1:
            return 'docs/figures/a1.5d0.4_final_test_pca_1shots.png'
        elif shot == 4:
            return 'docs/figures/4shot_baseline_pca.png'
        elif shot == 16:
            return 'docs/figures/baseline_pca.png'
        else:
            return 'docs/figures/32shot_baseline_pca.png'

def create_projection_grid(shot, mode='tsne'):
    # Mode: 'tsne' or 'pca'
    slots = [
        # (Title, image_path, banner_bg, banner_fg)
        (
            'Plain CLIP-LoRA (Baseline)',
            get_plain_lora_path(shot, mode),
            '#1A1A1A', '#FFFFFF'
        ),
        (
            'Run 1: CoOp Baseline',
            f'docs/figures2/run1_baseline_{shot}shot_test_{mode}.png',
            '#4C72B0', '#FFFFFF'
        ),
        (
            'Run 2: + Residual Class Tokens',
            f'docs/figures2/run2_learnable_tokens_{shot}shot_test_{mode}.png',
            '#DD8452', '#FFFFFF'
        ),
        (
            'Run 3: + Ordinal Cost Loss',
            f'docs/figures2/run3_ordinal_loss_{shot}shot_test_{mode}.png',
            '#55A868', '#FFFFFF'
        ),
        (
            'Run 4: + PromptSRC Reg.',
            f'docs/figures2/run4_promptsrc_{shot}shot_test_{mode}.png',
            '#C44E52', '#FFFFFF'
        ),
        (
            'Run 5: + Full Unified Method',
            f'docs/figures2/run5_full_method_{shot}shot_test_{mode}.png',
            '#8172B3', '#FFFFFF'
        ),
        (
            'Run 6: RT-LoRA (Ours)',
            f'docs/figures2/run6_rt_lora_{shot}shot_test_{mode}.png',
            '#E91E63', '#FFFFFF'
        ),
        (
            'Run 7: CSC + Vision LoRA',
            f'docs/figures2/run7_csc_vision_{shot}shot_test_{mode}.png',
            '#00897B', '#FFFFFF'
        ),
        (
            'Run 8: CSC + Dual LoRA',
            f'docs/figures2/run8_csc_both_{shot}shot_test_{mode}.png',
            '#D84315', '#FFFFFF'
        ),
        (
            'Run 9: CSC 100it + Vision LoRA',
            f'docs/figures2/run9_staged_csc_vision_{shot}shot_test_{mode}.png',
            '#0288D1', '#FFFFFF'
        ),
        (
            'Run 10: CSC 100it + Dual LoRA',
            f'docs/figures2/run10_staged_csc_both_{shot}shot_test_{mode}.png',
            '#6A1B9A', '#FFFFFF'
        ),
        (
            'Zero-Shot CLIP (Unadapted)',
            f'docs/figures/4shot_zs_{mode}.png',
            '#555555', '#FFFFFF'
        ),
    ]

    cols = 3
    rows = 4
    tile_w = 750
    tile_h = 565
    banner_h = 42

    grid_w = cols * tile_w
    grid_h = rows * tile_h
    grid_img = Image.new('RGB', (grid_w, grid_h), color='#FFFFFF')
    draw = ImageDraw.Draw(grid_img)
    font = ImageFont.truetype(font_bold_path, 22)

    for idx, (title, fpath, bg_col, fg_col) in enumerate(slots):
        r = idx // cols
        c = idx % cols
        x0 = c * tile_w
        y0 = r * tile_h

        # Load tile image
        if os.path.exists(fpath):
            tile = Image.open(fpath).convert('RGB')
            # resize tile to fit inside tile area minus banner
            tile_resized = tile.resize((tile_w, tile_h - banner_h), Image.Resampling.LANCZOS)
            grid_img.paste(tile_resized, (x0, y0 + banner_h))
        else:
            print(f'Warning: {fpath} not found for slot {title}')

        # Draw banner
        draw.rectangle([x0, y0, x0 + tile_w, y0 + banner_h], fill=bg_col)
        # Center title text
        bbox = font.getbbox(title)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = x0 + (tile_w - text_w) // 2
        text_y = y0 + (banner_h - text_h) // 2 - 2
        draw.text((text_x, text_y), title, font=font, fill=fg_col)

        # Draw border
        draw.rectangle([x0, y0, x0 + tile_w - 1, y0 + tile_h - 1], outline='#D0D0D0', width=1)

    out_name = f'comparison_{mode}_{shot}shot.png'
    out_path = os.path.join(OUT, out_name)
    grid_img.save(out_path, dpi=(200, 200), optimize=True)
    print(f'Saved {out_path} ({grid_img.size})')

def create_attention_progression():
    # 11 models x 4 shots
    models = [
        ('Plain CLIP-LoRA (Baseline)', '#1A1A1A', lambda s: f'docs/figures/attn_prog_{s}shot.png'),
        ('Run 1: CoOp Baseline', '#4C72B0', lambda s: f'docs/figures2/run1_baseline_{s}shot_attn2.png'),
        ('Run 2: + Residual Tokens', '#DD8452', lambda s: f'docs/figures2/run2_learnable_tokens_{s}shot_attn2.png'),
        ('Run 3: + Ordinal Loss', '#55A868', lambda s: f'docs/figures2/run3_ordinal_loss_{s}shot_attn2.png'),
        ('Run 4: + PromptSRC', '#C44E52', lambda s: f'docs/figures2/run4_promptsrc_{s}shot_attn2.png'),
        ('Run 5: + Full Unified', '#8172B3', lambda s: f'docs/figures2/run5_full_method_{s}shot_attn2.png'),
        ('Run 6: RT-LoRA (Ours)', '#E91E63', lambda s: f'docs/figures2/run6_rt_lora_{s}shot_attn2.png'),
        ('Run 7: CSC + Vision LoRA', '#00897B', lambda s: f'docs/figures2/run7_csc_vision_{s}shot_attn2.png'),
        ('Run 8: CSC + Dual LoRA', '#D84315', lambda s: f'docs/figures2/run8_csc_both_{s}shot_attn2.png'),
        ('Run 9: CSC (100it) + Vision', '#0288D1', lambda s: f'docs/figures2/run9_staged_csc_vision_{s}shot_attn2.png'),
        ('Run 10: CSC (100it) + Dual', '#6A1B9A', lambda s: f'docs/figures2/run10_staged_csc_both_{s}shot_attn2.png'),
    ]

    shots = [1, 4, 16, 32]
    shot_labels = ['1-Shot', '4-Shot', '16-Shot', '32-Shot']

    tile_w = 420
    tile_h = 160
    label_w = 260
    header_h = 44

    grid_w = label_w + len(shots) * tile_w
    grid_h = header_h + len(models) * tile_h

    grid_img = Image.new('RGB', (grid_w, grid_h), color='#FFFFFF')
    draw = ImageDraw.Draw(grid_img)
    font_head = ImageFont.truetype(font_bold_path, 20)
    font_lbl = ImageFont.truetype(font_bold_path, 17)

    # Draw header
    draw.rectangle([0, 0, label_w, header_h], fill='#2B2B2B')
    draw.text((15, 12), 'Model / Method', font=font_head, fill='#FFFFFF')

    for col_idx, (shot, lbl) in enumerate(zip(shots, shot_labels)):
        x0 = label_w + col_idx * tile_w
        draw.rectangle([x0, 0, x0 + tile_w, header_h], fill='#37474F')
        bbox = font_head.getbbox(lbl)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((x0 + (tile_w - tw) // 2, (header_h - th) // 2 - 2), lbl, font=font_head, fill='#FFFFFF')
        draw.rectangle([x0, 0, x0 + tile_w, header_h], outline='#607D8B', width=1)

    for row_idx, (mname, bg_col, get_path) in enumerate(models):
        y0 = header_h + row_idx * tile_h

        # Row label banner
        draw.rectangle([0, y0, label_w, y0 + tile_h], fill=bg_col)
        bbox = font_lbl.getbbox(mname)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((12, y0 + (tile_h - th) // 2 - 2), mname, font=font_lbl, fill='#FFFFFF')
        draw.rectangle([0, y0, label_w, y0 + tile_h], outline='#D0D0D0', width=1)

        # Tiles
        for col_idx, shot in enumerate(shots):
            x0 = label_w + col_idx * tile_w
            fpath = get_path(shot)
            if os.path.exists(fpath):
                tile = Image.open(fpath).convert('RGB')
                tile_resized = tile.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
                grid_img.paste(tile_resized, (x0, y0))
            else:
                print(f'Warning: {fpath} not found')
            draw.rectangle([x0, y0, x0 + tile_w, y0 + tile_h], outline='#E0E0E0', width=1)

    out_path = os.path.join(OUT, 'comparison_attention_progression.png')
    grid_img.save(out_path, dpi=(200, 200), optimize=True)
    print(f'Saved {out_path} ({grid_img.size})')

if __name__ == '__main__':
    for s in [1, 4, 16, 32]:
        create_projection_grid(s, 'tsne')
        create_projection_grid(s, 'pca')
    create_attention_progression()
