"""
MLLM Fine-Grained Prompt Ensembles for Few-Shot Fine-Grained Vision Classification.
Generated via UniFGVC CDV-Captioner (Category-Discriminative Visual Captioner) methodology
using Frontier Multimodal LLMs (Gemini / Qwen2.5-VL).
"""

PIAROM_SHAPE_PROMPTS = {
    # degree1 / grade 1 premium
    "degree1": [
        "a photo of a degree1 Piarom date fruit with long, slender cylindrical shape.",
        "a photo of an export-grade elongated Piarom date with straight symmetry and tight skin.",
        "a photo of a high aspect ratio grade 1 premium Piarom date with uniform tapered ends."
    ],
    "grade 1 premium": [
        "a photo of a degree1 Piarom date fruit with long, slender cylindrical shape.",
        "a photo of an export-grade elongated Piarom date with straight symmetry and tight skin.",
        "a photo of a high aspect ratio grade 1 premium Piarom date with uniform tapered ends."
    ],
    # degree2 / grade 2 standard
    "degree2": [
        "a photo of a degree2 Piarom date fruit with medium elongation and slight axis curvature.",
        "a photo of a grade 2 standard Piarom date with minor axis bending and slight skin separation.",
        "a photo of a moderately long Piarom date with slight lateral curve."
    ],
    "grade 2 standard": [
        "a photo of a degree2 Piarom date fruit with medium elongation and slight axis curvature.",
        "a photo of a grade 2 standard Piarom date with minor axis bending and slight skin separation.",
        "a photo of a moderately long Piarom date with slight lateral curve."
    ],
    # degree3 / grade 3
    "degree3": [
        "a photo of a degree3 Piarom date fruit with pronounced curvature and irregular body contour.",
        "a photo of a grade 3 asymmetrical Piarom date with shorter relative length.",
        "a photo of a third-grade Piarom date with uneven wrinkled surface and noticeable bend."
    ],
    "grade 3": [
        "a photo of a degree3 Piarom date fruit with pronounced curvature and irregular body contour.",
        "a photo of a grade 3 asymmetrical Piarom date with shorter relative length.",
        "a photo of a third-grade Piarom date with uneven wrinkled surface and noticeable bend."
    ],
    # kade / small stunted
    "kade": [
        "a photo of a kade Piarom date fruit with stunted, abnormally short length.",
        "a photo of a stubby small stunted Piarom date with low length-to-width aspect ratio.",
        "a photo of an abnormally round, short Piarom date fruit lacking typical elongation."
    ],
    "small stunted": [
        "a photo of a kade Piarom date fruit with stunted, abnormally short length.",
        "a photo of a stubby small stunted Piarom date with low length-to-width aspect ratio.",
        "a photo of an abnormally round, short Piarom date fruit lacking typical elongation."
    ],
    # lehide / crushed bruised
    "lehide": [
        "a photo of a lehide Piarom date fruit with crushed flattened body and mechanically deformed tissue.",
        "a photo of a bruised, sunken Piarom date with collapsed structure and ruptured skin.",
        "a photo of a mechanically compressed crushed bruised Piarom date."
    ],
    "crushed bruised": [
        "a photo of a lehide Piarom date fruit with crushed flattened body and mechanically deformed tissue.",
        "a photo of a bruised, sunken Piarom date with collapsed structure and ruptured skin.",
        "a photo of a mechanically compressed crushed bruised Piarom date."
    ]
}

WALNUT_COLOR_PROMPTS = {
    # lux / luxury extra light
    "lux": [
        "a photo of a lux walnut kernel with very bright ivory-white color and high luminance.",
        "a photo of a luxury extra light walnut kernel with clean uniform light pellicle.",
        "a photo of an extra light high-grade walnut meat free from dark discoloration."
    ],
    "luxury extra light": [
        "a photo of a lux walnut kernel with very bright ivory-white color and high luminance.",
        "a photo of a luxury extra light walnut kernel with clean uniform light pellicle.",
        "a photo of an extra light high-grade walnut meat free from dark discoloration."
    ],
    # white momtaz / premium white
    "white momtaz": [
        "a photo of a white momtaz walnut kernel with bright creamy-white pellicle and golden undertone.",
        "a photo of a premium white walnut kernel with uniform light surface color across lobes.",
        "a photo of a high-quality white momtaz walnut meat with clean light-tan ridges."
    ],
    "premium white": [
        "a photo of a white momtaz walnut kernel with bright creamy-white pellicle and golden undertone.",
        "a photo of a premium white walnut kernel with uniform light surface color across lobes.",
        "a photo of a high-quality white momtaz walnut meat with clean light-tan ridges."
    ],
    # white mamooli / standard white
    "white mamooli": [
        "a photo of a white mamooli walnut kernel with standard light-brown tan color.",
        "a photo of a standard white commercial-grade walnut kernel with slightly darker shade in crevices.",
        "a photo of a light amber-tan white mamooli walnut meat."
    ],
    "standard white": [
        "a photo of a white mamooli walnut kernel with standard light-brown tan color.",
        "a photo of a standard white commercial-grade walnut kernel with slightly darker shade in crevices.",
        "a photo of a light amber-tan white mamooli walnut meat."
    ],
    # brown plus / high-quality brown
    "brown plus": [
        "a photo of a brown plus walnut kernel with warm light-amber and golden-brown hue.",
        "a photo of a high-quality brown walnut kernel with moderate color saturation across convolutions.",
        "a photo of a light-brown amber plus walnut meat."
    ],
    "high-quality brown": [
        "a photo of a brown plus walnut kernel with warm light-amber and golden-brown hue.",
        "a photo of a high-quality brown walnut kernel with moderate color saturation across convolutions.",
        "a photo of a light-brown amber plus walnut meat."
    ],
    # brown momtaz / premium brown
    "brown momtaz": [
        "a photo of a brown momtaz walnut kernel with dark brown exterior pellicle contrasting with lighter meat at edges.",
        "a photo of a premium brown walnut kernel with two-tone contrast between outer skin and fractured interior.",
        "a photo of an amber-brown momtaz walnut meat with visible light kernel core."
    ],
    "premium brown": [
        "a photo of a brown momtaz walnut kernel with dark brown exterior pellicle contrasting with lighter meat at edges.",
        "a photo of a premium brown walnut kernel with two-tone contrast between outer skin and fractured interior.",
        "a photo of an amber-brown momtaz walnut meat with visible light kernel core."
    ],
    # siah goshti / dark meaty
    "siah goshti": [
        "a photo of a siah goshti walnut kernel with very dark blackish-brown to charcoal pellicle.",
        "a photo of a dark meaty walnut kernel with deeply pigmented oxidized meat.",
        "a photo of a black-brown siah goshti walnut meat with dark shading throughout the lobes."
    ],
    "dark meaty": [
        "a photo of a siah goshti walnut kernel with very dark blackish-brown to charcoal pellicle.",
        "a photo of a dark meaty walnut kernel with deeply pigmented oxidized meat.",
        "a photo of a black-brown siah goshti walnut meat with dark shading throughout the lobes."
    ]
}


def get_mllm_prompts(dataset_name, classnames=None):
    """
    Returns MLLM prompt ensemble dictionary for the target dataset if available.
    """
    d_clean = dataset_name.lower().replace("-", "_").replace(" ", "_")
    if "piarom_shape" in d_clean or ("piarom" in d_clean and "shape" in d_clean):
        return PIAROM_SHAPE_PROMPTS
    elif "walnut" in d_clean:
        return WALNUT_COLOR_PROMPTS
    return None
