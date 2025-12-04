import os


class CampbellDataset:
    """
    Dataset loader for the campbell commentary from the GT-commentaries-OCR dataset
    """

    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.keys = sorted([
            fname.replace(".png", "") for fname in os.listdir(self.root_dir) if fname.endswith(".png")
        ])

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]

        img_path = os.path.join(self.root_dir, f"{key}.png")
        gt_path = os.path.join(self.root_dir, f"{key}.gt.txt")


        with open(gt_path, "r", encoding="utf-8") as f:
            gt_text = f.read().strip()


        return img_path, gt_text