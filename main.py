from cb_dataloader import CampbellDataset
from torch.utils.data import DataLoader

dataset = CampbellDataset("Dataset/GT-pairs")

loader = DataLoader(
    dataset,
    batch_size = 1,
    shuffle = False,
    num_workers = 0
)
