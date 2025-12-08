import os
from PIL import Image
from cb_dataloader import CampbellDataset

num_images = 5

def jpg_modify():

    dataset_dir = "Dataset/GT-pairs"
    output_dir = "Dataset/jpeg_folder"

    # Load dataset
    ds = CampbellDataset(dataset_dir)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"Total dataset size: {len(ds)}")
    print(f"Processing first {num_images} images")

    for i in range(num_images):
        sample = ds[i]

        img_id = sample["id"]
        img = sample["image"]     # PIL Image
        text = sample["text"]

        img.load()

        out_img_path = os.path.join(output_dir, f"{img_id}.jpg")
        out_gt_path = os.path.join(output_dir, f"{img_id}.gt.txt")

        # Handle PNG transparency
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            background.save(out_img_path, "JPEG", quality=10)

        else:
            rgb_img = img.convert("RGB")
            rgb_img.save(out_img_path, "JPEG", quality=10)

        # Write ground-truth text
        with open(out_gt_path, "w", encoding="utf-8") as f:
            f.write(text)

def main(): 
    jpg_modify()

if __name__ == "__main__": 
    main()