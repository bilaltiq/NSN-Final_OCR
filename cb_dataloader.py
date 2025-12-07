import os
from torch.utils.data import Dataset
from PIL import Image

class CampbellDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.ids = sorted([
            fname.replace(".png", "")
            for fname in os.listdir(self.root_dir)
            if fname.endswith(".png")
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]

        img_path = os.path.join(self.root_dir, sample_id + ".png")
        txt_path = os.path.join(self.root_dir, sample_id + ".gt.txt")

        # Load image in PIL
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
