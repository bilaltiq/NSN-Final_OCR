from cb_dataloader import CampbellDataset

ds = CampbellDataset("Dataset/GT-pairs")

print("Total samples:", len(ds))

img_path, gt = ds[0]

print("Image:", img_path)
print("GT text:", gt)