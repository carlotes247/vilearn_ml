import glob
import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from training.model_cfg import RANDOM_SEED, ROOT_DIR, TRAIN_SIZE, CONTEXT_WINDOW, FEATURE_PATH
from utils.audio_feature_utils import compute_context

def split_data_set(feature_path: str, data_set: str) -> list:
    """
    Split the data set into train, validation, and test samples.

    Args:
        feature_path: A string representing the directory of the data set.
        data_set: A string representing the data set to split ('train', 'val' or 'test').

    Returns:
        The list of samples for the specified data set.
    """
    samples = [path for path in glob.glob(f'{ROOT_DIR}/{feature_path}/**/*.csv')]

    train_samples, test_samples = train_test_split(samples, train_size=TRAIN_SIZE, random_state=RANDOM_SEED)
    if data_set == 'train':
        return train_samples
    val_samples, test_samples = train_test_split(test_samples, test_size=0.5, random_state=RANDOM_SEED)
    if data_set == 'val':
        return val_samples
    return test_samples


class MyDataset(Dataset):
    def __init__(self, feature_path: str, root_dir: str, spectral_features: bool,
                 emotional_speech_features: bool, add_context: bool,
                 data_set: str, context_window: int = CONTEXT_WINDOW,
                 transform=None, target_transform=None):
        if data_set not in ['train', 'val', 'test']:
            Exception(f'data_set must be "train", "val", "test" or "new_voice", got {data_set}')
        self.feature_path = feature_path
        self.root_dir = root_dir
        self.spectral_features = spectral_features,
        self.emotional_speech_features = emotional_speech_features,
        self.add_context = add_context,
        self.data_set = data_set,
        self.context_window = context_window,
        self.transform = transform
        self.target_transform = target_transform

        self.data = []

        samples = [Path(path).absolute() for path in split_data_set(feature_path, data_set)]
        # set up loading bar
        pbar = tqdm(total=len(samples), desc='Loading data')

        mean_face = []
        for sample in samples:
            sample = str(sample)

            # TODO: load data into
            input_features=None
            classification_gt=None
            # TODO: ....


            if add_context:
                input_features = compute_context(input_features, context_window)

            self.data.append((classification_gt, input_features))
            pbar.update()

        pbar.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        classification_gt, input_features = self.data[idx]

        if self.transform:
            classification_gt = self.transform(classification_gt)
        if self.target_transform:
            input_features = self.target_transform (input_features)

        classification_gt = torch.from_numpy(classification_gt).float()
        input_features = torch.from_numpy(input_features).float()
        return classification_gt, input_features


def main():
    # compute mean face
    train_set = MyDataset(feature_path=FEATURE_PATH, root_dir=ROOT_DIR,
                          spectral_features=True, emotional_speech_features=False, remove_identity=False,
                          add_context=False, data_set='train', )


if __name__ == "__main__":
    main()
