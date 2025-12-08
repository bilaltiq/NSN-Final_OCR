import os
import random
import numpy as np
from PIL import Image
from cb_dataloader import CampbellDataset

num_images = 5
noise_intensity = 0.1
color_noise_factor = 0.5


def add_iso_noise(image, noise_intensity, color_noise_factor):
    # Convert PIL Image to NumPy
    img_pil = image.convert("RGB")   # FIXED
    img_np = np.array(img_pil) / 255.0

    # Luminance noise (grayscale noise expanded to 3 channels)
    luminance_noise = np.random.normal(0, noise_intensity, img_np.shape[:2])
    luminance_noise = np.expand_dims(luminance_noise, axis=2)

    # Color noise (RGB channels)
    color_noise = np.random.normal(0, noise_intensity * color_noise_factor, img_np.shape)

    # Combine noise
    noisy_img_np = img_np + luminance_noise + color_noise
    noisy_img_np = np.clip(noisy_img_np, 0, 1)

    # Convert back to PIL
    noisy_img_np = (noisy_img_np * 255).astype(np.uint8)
    noisy_img_pil = Image.fromarray(noisy_img_np)

    return noisy_img_pil


def modify():
    dataset_dir = "Dataset/GT-pairs"      
    output_dir = "Dataset/iso_noise_folder"  

    # Load dataset
    ds = CampbellDataset(dataset_dir)

    # Make output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Total dataset size: {len(ds)}")
    print(f"Processing first {num_images} images")

    for i in range(num_images):
        sample = ds[i]

        img_id = sample["id"]
        img = sample["image"]   # PIL Image
        text = sample["text"]

        # Apply noise
        iso_img = add_iso_noise(img, noise_intensity, color_noise_factor)

        # Output file paths
        out_img_path = os.path.join(output_dir, f"{img_id}_iso.png")
        out_gt_path = os.path.join(output_dir, f"{img_id}_iso.gt.txt")

        iso_img.save(out_img_path)

        with open(out_gt_path, "w", encoding="utf-8") as f:
            f.write(text)

def main(): 
    modify()

if __name__ == "__main__":
    main()