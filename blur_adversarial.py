import os
import random
from PIL import ImageFilter
from cb_dataloader import CampbellDataset

num_images = 5

def add_blur(img):
    blurred_image  = img.filter(ImageFilter.GaussianBlur(radius=random.randint(0, 3)))
    return blurred_image
    

def modify():
    ds = CampbellDataset("Dataset/GT-pairs")
    output_dir = "Dataset/BlurGT-Pairs"

    os.makedirs(output_dir, exist_ok = True)

    print(f"Total dataset size: {len(ds)}")
    print(f"Processing first {num_images} images")

    for i in range(num_images):
        sample = ds[i]
        img_id = sample["id"]
        img = sample["image"]
        text = sample["text"]

        blurred_img = add_blur(img)
        out_img_path = os.path.join(output_dir, f"{img_id}_blur.png")
        blurred_img.save(out_img_path)

        out_gt_path = os.path.join(output_dir, f"{img_id}_blur.gt.txt")
        with open(out_gt_path, "w", encoding="utf-8") as f:
            f.write(text)


def main():
    modify()


if __name__ == "__main__":
    main()
