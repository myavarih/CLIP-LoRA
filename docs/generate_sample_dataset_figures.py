#!/usr/bin/env python3
"""
Generate publication-quality sample image grids and individual strips for
Walnut and Piarom Shape datasets, adhering to the requested format:
'image: <class> (<translation>)'
"""

import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

# Canonical dataset definitions and semantic translations
WALNUT_CONFIG = {
    'title': 'Walnut Quality Grading Dataset',
    'dir_name': 'Walnut_Color_Parvizi_3',
    'out_prefix': 'walnut',
    'classes': [
        ('siah goshti', 'dark meaty', 'سیاه گوشتی'),
        ('brown plus', 'high-quality brown', 'قهوه‌ای پلاس'),
        ('brown momtaz', 'premium brown', 'قهوه‌ای ممتاز'),
        ('white mamooli', 'standard white', 'سفید معمولی'),
        ('white momtaz', 'premium white', 'سفید ممتاز'),
        ('lux', 'luxury extra light', 'لوکس'),
    ]
}

PIAROM_CONFIG = {
    'title': 'Piarom Date Shape Grading Dataset',
    'dir_name': 'Piarom_Shape_Ameri_3',
    'out_prefix': 'piarom_shape',
    'classes': [
        ('degree1', 'grade 1 premium', 'درجه ۱ ممتاز'),
        ('degree2', 'grade 2 standard', 'درجه ۲ استاندارد'),
        ('degree3', 'grade 3', 'درجه ۳'),
        ('kade', 'small stunted', 'کده (ریز / کوتاه)'),
        ('lehide', 'crushed bruised', 'لهیده (آسیب‌دیده)'),
    ]
}

def get_fonts():
    bold_path = fm.findfont('DejaVu Sans:weight=bold')
    reg_path = fm.findfont('DejaVu Sans')
    return bold_path, reg_path

def letterbox_square(im, target_size=180, bg_color=(24, 26, 32)):
    """SquarePad preserving aspect ratio centered in target_size x target_size."""
    w, h = im.size
    scale = min((target_size - 12) / w, (target_size - 12) / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    square = Image.new('RGB', (target_size, target_size), bg_color)
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    square.paste(resized, (paste_x, paste_y))
    return square

def select_8_samples(class_dir):
    files = sorted([f for f in os.listdir(class_dir) if f.endswith('.bmp') and not f.startswith('.')])
    if len(files) <= 8:
        return files
    # Pick 8 diverse, evenly-spaced images across the split
    indices = [int(i * (len(files) - 1) / 7.0) for i in range(8)]
    return [files[idx] for idx in indices]

def create_dataset_grid(config, repo_root, out_dir):
    bold_path, reg_path = get_fonts()
    font_title = ImageFont.truetype(bold_path, 28)
    font_subtitle = ImageFont.truetype(reg_path, 16)
    font_row_cls = ImageFont.truetype(bold_path, 18)
    font_row_trans = ImageFont.truetype(reg_path, 15)
    font_img_label = ImageFont.truetype(reg_path, 12)

    tile_size = 180
    header_w = 260
    pad_x = 14
    pad_y = 14
    margin_top = 100
    margin_bottom = 25
    margin_side = 25

    num_cols = 8
    num_rows = len(config['classes'])

    total_w = margin_side * 2 + header_w + num_cols * tile_size + (num_cols - 1) * pad_x
    row_h = tile_size + 30
    total_h = margin_top + num_rows * row_h + (num_rows - 1) * pad_y + margin_bottom

    canvas = Image.new('RGB', (total_w, total_h), (15, 17, 21))
    draw = ImageDraw.Draw(canvas)

    # Title Banner
    draw.text((margin_side, 22), config['title'], fill=(245, 245, 245), font=font_title)
    sub_text = f"Covering all {num_rows} classes — 8 sample images per class — Format: image: class (translation)"
    draw.text((margin_side, 60), sub_text, fill=(160, 165, 175), font=font_subtitle)
    draw.line([(margin_side, 88), (total_w - margin_side, 88)], fill=(45, 50, 60), width=2)

    data_dir = os.path.join(repo_root, 'FewShotData', config['dir_name'], 'test')

    for r_idx, (cls, trans, fa_trans) in enumerate(config['classes']):
        class_path = os.path.join(data_dir, cls)
        sample_files = select_8_samples(class_path)

        y_top = margin_top + r_idx * (row_h + pad_y)

        # Draw row header box
        header_rect = [margin_side, y_top, margin_side + header_w - 15, y_top + row_h]
        draw.rectangle(header_rect, fill=(24, 28, 36), outline=(48, 54, 68), width=1)
        
        # Row header text
        draw.text((margin_side + 14, y_top + 18), "Class:", fill=(140, 150, 165), font=font_img_label)
        draw.text((margin_side + 14, y_top + 34), cls, fill=(255, 215, 110), font=font_row_cls)
        
        draw.text((margin_side + 14, y_top + 70), "Translation:", fill=(140, 150, 165), font=font_img_label)
        draw.text((margin_side + 14, y_top + 88), f"({trans})", fill=(130, 210, 255), font=font_row_trans)
        
        draw.text((margin_side + 14, y_top + 130), f"Format:", fill=(140, 150, 165), font=font_img_label)
        draw.text((margin_side + 14, y_top + 148), f"image: {cls}", fill=(200, 205, 215), font=font_img_label)
        draw.text((margin_side + 14, y_top + 164), f"({trans})", fill=(170, 175, 185), font=font_img_label)

        # Draw 8 image tiles
        for c_idx, fname in enumerate(sample_files):
            img_path = os.path.join(class_path, fname)
            raw_im = Image.open(img_path)
            tile = letterbox_square(raw_im, target_size=tile_size)

            x_left = margin_side + header_w + c_idx * (tile_size + pad_x)
            
            # Card background
            card_rect = [x_left - 2, y_top - 2, x_left + tile_size + 2, y_top + tile_size + 2]
            draw.rectangle(card_rect, fill=(30, 35, 45), outline=(50, 56, 70), width=1)
            
            # Paste tile
            canvas.paste(tile, (x_left, y_top))

            # Tile caption: image # and class (translation)
            cap_text = f"image: {cls} #{c_idx+1}"
            draw.text((x_left + 4, y_top + tile_size + 6), cap_text, fill=(185, 192, 205), font=font_img_label)

    out_file = os.path.join(out_dir, f"{config['out_prefix']}_sample_images_grid.png")
    canvas.save(out_file, quality=95)
    print(f"Saved: {out_file}")
    return out_file

def create_individual_strips(config, repo_root, out_dir):
    """Creates a 1x8 horizontal strip image for each class individually."""
    bold_path, reg_path = get_fonts()
    font_title = ImageFont.truetype(bold_path, 22)
    font_sub = ImageFont.truetype(reg_path, 15)
    font_lbl = ImageFont.truetype(reg_path, 12)

    tile_size = 180
    pad_x = 12
    margin_side = 20
    margin_top = 70
    margin_bottom = 35

    num_cols = 8
    total_w = margin_side * 2 + num_cols * tile_size + (num_cols - 1) * pad_x
    total_h = margin_top + tile_size + margin_bottom

    data_dir = os.path.join(repo_root, 'FewShotData', config['dir_name'], 'test')
    strip_dir = os.path.join(out_dir, 'class_strips')
    os.makedirs(strip_dir, exist_ok=True)

    for cls, trans, fa_trans in config['classes']:
        canvas = Image.new('RGB', (total_w, total_h), (16, 18, 23))
        draw = ImageDraw.Draw(canvas)

        # Strip header
        draw.text((margin_side, 14), f"image: {cls} ({trans})", fill=(255, 215, 110), font=font_title)
        draw.text((margin_side, 42), f"Dataset: {config['title']} | 8 Test Samples", fill=(150, 160, 175), font=font_sub)
        draw.line([(margin_side, 62), (total_w - margin_side, 62)], fill=(45, 50, 60), width=1)

        class_path = os.path.join(data_dir, cls)
        sample_files = select_8_samples(class_path)

        for c_idx, fname in enumerate(sample_files):
            img_path = os.path.join(class_path, fname)
            raw_im = Image.open(img_path)
            tile = letterbox_square(raw_im, target_size=tile_size)

            x_left = margin_side + c_idx * (tile_size + pad_x)
            y_top = margin_top

            card_rect = [x_left - 1, y_top - 1, x_left + tile_size + 1, y_top + tile_size + 1]
            draw.rectangle(card_rect, fill=(28, 32, 42), outline=(50, 56, 72), width=1)
            canvas.paste(tile, (x_left, y_top))

            draw.text((x_left + 4, y_top + tile_size + 6), f"image {c_idx+1}: {cls}", fill=(180, 190, 205), font=font_lbl)

        clean_cls_name = cls.replace(' ', '_')
        strip_file = os.path.join(strip_dir, f"{config['out_prefix']}_{clean_cls_name}_strip.png")
        canvas.save(strip_file, quality=95)
        print(f"Saved strip: {strip_file}")

def export_individual_pngs(config, repo_root, out_dir):
    """Saves each of the 8 samples per class as clean PNGs."""
    data_dir = os.path.join(repo_root, 'FewShotData', config['dir_name'], 'test')
    png_base = os.path.join(out_dir, 'individual_samples', config['out_prefix'])

    for cls, trans, fa_trans in config['classes']:
        clean_cls = cls.replace(' ', '_')
        cls_out = os.path.join(png_base, clean_cls)
        os.makedirs(cls_out, exist_ok=True)

        class_path = os.path.join(data_dir, cls)
        sample_files = select_8_samples(class_path)

        for idx, fname in enumerate(sample_files):
            raw_im = Image.open(os.path.join(class_path, fname))
            sq = letterbox_square(raw_im, target_size=224)
            dst_name = f"{config['out_prefix']}_{clean_cls}_pic{idx+1}.png"
            sq.save(os.path.join(cls_out, dst_name))

if __name__ == '__main__':
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    out_dir = os.path.join(repo_root, 'docs', 'figures', 'dataset_samples')
    os.makedirs(out_dir, exist_ok=True)

    print("Generating Walnut sample images...")
    create_dataset_grid(WALNUT_CONFIG, repo_root, out_dir)
    create_individual_strips(WALNUT_CONFIG, repo_root, out_dir)
    export_individual_pngs(WALNUT_CONFIG, repo_root, out_dir)

    print("\nGenerating Piarom Shape sample images...")
    create_dataset_grid(PIAROM_CONFIG, repo_root, out_dir)
    create_individual_strips(PIAROM_CONFIG, repo_root, out_dir)
    export_individual_pngs(PIAROM_CONFIG, repo_root, out_dir)

    print("\nAll sample pictures generated successfully!")
