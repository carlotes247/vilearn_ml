from torch.utils.data import DataLoader
from torch_vilearn.torch_group_dataset import TorchGroupDataset

class TorchGroupDataLoader():
    group_dataloader: DataLoader
    def __init__(self, groupDataset: TorchGroupDataset):
        try:
            self.group_dataloader = DataLoader(dataset=groupDataset, batch_size=4, shuffle=True , num_workers=2)
        except Exception as err:
            print(f"Error: Data Loader cannot be created with exception {err=}, {type(err)=}")