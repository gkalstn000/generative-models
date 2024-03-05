import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from .dataset import StableDataModuleFromConfig
from omegaconf import DictConfig
from typing import Optional
import albumentations
import numpy as np
import cv2
import json
import os
from PIL import Image
import random
random.seed(10)
class ConnectbrickDataDictWrapper(Dataset):
    def __init__(self, dset, config):
        super().__init__()
        self.dset = dset
        self.config = config
        self.size = config.loader.size



    def __getitem__(self, i):
        x, y = self.dset[i]
        root = self.config.datapipeline.urls
        image = Image.open(os.path.join(root, 'images', x)).convert("RGB")
        image = image.resize((self.size, self.size), Image.BILINEAR)
        image = np.array(image).astype(np.uint8)

        image = (image/127.5 - 1.0).astype(np.float32)
        image = image.transpose((2, 0, 1))

        return {"jpg": image, "txt": y}

    def __len__(self):
        return len(self.dset)


class ConnectbrickLoader(StableDataModuleFromConfig):
    def __init__(
        self,
        train: DictConfig,
        validation: Optional[DictConfig] = None,
        test: Optional[DictConfig] = None,
        skip_val_loader: bool = False,
        dummy: bool = False,
    ):
        super(ConnectbrickLoader, self).__init__(train, validation, test, skip_val_loader, dummy)

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * 2.0 - 1.0)]
        )

        self.batch_size = train.loader.batch_size
        self.num_workers = train.loader.num_workers
        self.prefetch_factor = train.loader.prefetch_factor if self.num_workers > 0 else 0
        self.shuffle = train.loader.shuffle

    def prepare_data(self):
        root = self.train_config.datapipeline.urls
        with open(os.path.join(root, 'meta.json'), 'r') as file:
            data = json.load(file)
        caption = [c for c in data['captions'] if c]
        images = data['images'][:len(caption)]
        dataset = [[i, c] for i, c in zip(images, caption)]
        random.shuffle(dataset)
        num_val = 20

        train, test = dataset[num_val:], dataset[:num_val]

        self.train_dataset = ConnectbrickDataDictWrapper(train, self.train_config)
        self.test_dataset = ConnectbrickDataDictWrapper(test, self.train_config)




    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )


if __name__ == "__main__":
    dset = ConnectbrickLoader(
        torchvision.datasets.MNIST(
            root=".data/",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Lambda(lambda x: x * 2.0 - 1.0)]
            ),
        )
    )
    ex = dset[0]
