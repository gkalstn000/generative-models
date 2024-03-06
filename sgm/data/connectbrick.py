import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from .dataset import StableDataModuleFromConfig
from omegaconf import DictConfig
from typing import Optional
import albumentations
import torch
import numpy as np
import cv2
import json
import os
from PIL import Image
import random
random.seed(10)
class ConnectbrickDataDictWrapper(Dataset):
    def __init__(self, dset, config, center_crop=True, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05):
        super().__init__()
        self.dset = dset
        self.config = config
        self.center_crop = center_crop
        self.t_drop_rate = t_drop_rate
        self.i_drop_rate = i_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.center_crop = center_crop
        self.size = config.loader.size

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def __getitem__(self, i):
        image_file, text = self.dset[i]

        # read image
        root = self.config.datapipeline.urls
        raw_image = Image.open(os.path.join(root, 'images', image_file)).convert("RGB")

        # original size
        original_width, original_height = raw_image.size
        original_size = torch.tensor([original_height, original_width])

        image_tensor = self.transform(raw_image.convert("RGB"))
        # random crop
        delta_h = image_tensor.shape[1] - self.size
        delta_w = image_tensor.shape[2] - self.size
        assert not all([delta_h, delta_w])

        if self.center_crop:
            top = delta_h // 2
            left = delta_w // 2
        else:
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        image = transforms.functional.crop(
            image_tensor, top=top, left=left, height=self.size, width=self.size
        )
        crop_coords_top_left = torch.tensor((top, left))



        return {
            "image": image,
            "txt": text,
            "original_size_as_tuple": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size_as_tuple": torch.tensor((self.size, self.size)),
        }




        image = Image.open(os.path.join(root, 'images', x)).convert("RGB")
        original_size = image.size[::-1] # (h, w)

        image = image.resize((self.size, self.size), Image.BILINEAR)
        image = np.array(image).astype(np.uint8)

        image = (image/127.5 - 1.0).astype(np.float32)
        image = image.transpose((2, 0, 1))

        return {"jpg": image,
                "txt": y,
                'original_size_as_tuple': original_size}

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
