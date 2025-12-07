import os
import random
from PIL import Image, ImageDraw, ImageFilter
from cb_dataloader import CampbellDataset


num_images = 5

def add_shadow (img):
    width, height = img.size

    shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow)

    pts = [
        (random.randint(0, width//3), random.randint(0, height)),
        (random.randint(width//3, 2*width//3), random.randint(0, height)),
        (random.randint(2*width//3, width), random.randint(0, height))
    ]

    darkness = random.randint(110, 180)

    draw.polygon(pts, fill=(0, 0, 0, darkness))

    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=random.randint(15, 35)))

    #shadow ovr image
    img_shadow = Image.alpha_composite(img.convert("RGBA"), shadow)

    return img_shadow.convert("RGB")

def modify():
    ds = CampbellDataset("NSN-Final_OCR/Dataset/GT-pairs")
    output_dir = "NSN-Final_OCR/Dataset/ShadowedGT-Pairs"

    os.makedirs(output_dir, exist_ok = True)

    print(f"Total dataset size: {len(ds)}")
    print(f"Processing first {num_images} images")

    for i in range(num_images):
        sample = ds[i]
        img_id = sample["id"]
        img = sample["image"]
        text = sample["text"]

        shadow_img = add_shadow(img)

        out_img_path = os.path.join(output_dir, f"{img_id}_shadow.png")
        shadow_img.save(out_img_path)

        out_gt_path = os.path.join(output_dir, f"{img_id}_shadow.gt.txt")
        with open(out_gt_path, "w", encoding="utf-8") as f:
            f.write(text)


def main():
    modify()


if __name__ == "__main__":
    main()
