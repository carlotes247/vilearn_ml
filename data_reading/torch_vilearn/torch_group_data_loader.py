from torch.utils.data import DataLoader
from torch_group_dataset import TorchGroupDataset

class TorchGroupDataLoader():
    group_dataloader: DataLoader
    def __init__(self, groupDataset: TorchGroupDataset):
        self.group_dataloader = DataLoader(dataset=groupDataset, batch_size=4, shuffle=True , num_workers=2)