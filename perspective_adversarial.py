import os
import random
from PIL import Image, ImageDraw, ImageFilter
from cb_dataloader import CampbellDataset

num_images = 5

def add_perspective_shift (img):
    angle = random.uniform(-4, 4)

    rotated_img = img.rotate(angle, expand = True)

    return rotated_img
    

def modify():
    ds = CampbellDataset("NSN-Final_OCR/Dataset/GT-pairs")
    output_dir = "NSN-Final_OCR/Dataset/PerspectiveGT-Pairs"

    os.makedirs(output_dir, exist_ok = True)

    print(f"Total dataset size: {len(ds)}")
    print(f"Processing first {num_images} images")

    for i in range(num_images):
        sample = ds[i]
        img_id = sample["id"]
        img = sample["image"]
        text = sample["text"]

        shifted_img = add_perspective_shift(img)

        out_img_path = os.path.join(output_dir, f"{img_id}_perspective.png")
        shifted_img.save(out_img_path)

        out_gt_path = os.path.join(output_dir, f"{img_id}_perspective.gt.txt")
        with open(out_gt_path, "w", encoding="utf-8") as f:
            f.write(text)


def main():
    modify()


if __name__ == "__main__":
    main()
