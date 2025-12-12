import os
from torch.utils.data import Dataset
from PIL import Image

class CampbellDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # pick up *.bin.png
        self.ids = sorted([
            fname.replace(".bin.png", "")
            for fname in os.listdir(self.root_dir)
            if fname.endswith(".bin.png")
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]

        img_path = os.path.join(self.root_dir, sample_id + ".bin.png")
        txt_path = os.path.join(self.root_dir, sample_id + ".gt.txt")  # <- fixed

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Load GT text
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        return {
            "id": sample_id,
            "image": image,
            "text": text
        }
