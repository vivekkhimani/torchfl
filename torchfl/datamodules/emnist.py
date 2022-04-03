#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""PyTorch LightningDataModule for EMNIST (balanced, byclass, bymerge, digits, letters, mnist) dataset.

Raises:
    - ValueError: The given dataset name is not supported. Supported: balanced, byclass, bymerge, digits, letters, mnist.

Returns:
    - DatasetSplit: Implementation of PyTorch key-value based Dataset.
    - EMNISTDataModule: PyTorch LightningDataModule for EMNIST datasets. Supports iid and non-iid splits.
"""
import torch
import numpy as np
import pytorch_lightning as pl
from typing import Set, Type, Literal, Iterable, Tuple, Any, Optional, Dict, List
import os
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision.datasets import EMNIST
from torchvision import transforms

torch.manual_seed(42)
np.random.seed(42)
pl.seed_everything(42)

###################
# Begin Constants #
###################

SUPPORTED_DATASETS: Set[str] = {
    "balanced",
    "byclass",
    "bymerge",
    "digits",
    "letters",
    "mnist",
}
SUPPORTED_DATASETS_LITERAL: Type[  # type: ignore
    Literal["balanced", "byclass", "bymerge", "digits", "letters", "mnist"]
] = Literal[
    "balanced", "byclass", "bymerge", "digits", "letters", "mnist"
]  # type: ignore

DEFAULT_TRANSFORMS: transforms.Compose = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
#################
# End Constants #
#################


class DatasetSplit(Dataset):
    """Implementation of PyTorch key-value based Dataset."""

    def __init__(self, dataset: Dataset, idxs: Iterable[int]) -> None:
        """Constructor

        Args:
            - dataset (Dataset): PyTorch Dataset.
            - idxs (Iterable[int]): collection of indices.
        """
        self.dataset: Dataset = dataset
        self.idxs: Iterable[int] = list(idxs)
        all_targets: np.ndarray = (
            np.array(dataset.targets)
            if isinstance(dataset.targets, list)
            else dataset.targets.numpy()
        )
        self.targets: np.ndarray = all_targets[self.idxs]

    def __len__(self) -> int:
        """Overriding the length method.

        Returns:
            - int: length of the collection of indices.
        """
        return len(self.idxs)  # type: ignore

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Overriding the get method.

        Args:
            - index (int): index for querying.

        Returns:
            - Tuple[Any, Any]: returns the key-value pair as a tuple.
        """
        image, label = self.dataset[self.idxs[index]]  # type: ignore
        return image, label


class EMNISTDataModule(pl.LightningDataModule):
    """PyTorch LightNingDataModule for CIFAR datasets."""

    def __init__(
        self,
        data_dir: str = os.path.join(os.pardir, "data"),
        dataset_name: SUPPORTED_DATASETS_LITERAL = "mnist",  # type: ignore
        validation_split: float = 0.1,
        train_batch_size: int = 32,
        validation_batch_size: int = 32,
        test_batch_size: int = 32,
        predict_batch_size: int = 32,
        train_transforms: transforms.Compose = DEFAULT_TRANSFORMS,
        val_transforms: transforms.Compose = DEFAULT_TRANSFORMS,
        test_transforms: transforms.Compose = DEFAULT_TRANSFORMS,
        predict_transforms: transforms.Compose = DEFAULT_TRANSFORMS,
    ):  # type: ignore
        """Constructor

        Args:
            - data_dir (str, optional): Default directory to download the dataset to. Defaults to os.pardir.
            - dataset_name (SUPPORTED_DATASETS_LITERAL, optional): Name of the dataset to be used. Defaults to "mnist".
            - validation_split (float, optional): Fraction of training images to be used as validation. Defaults to 0.1.
            - train_batch_size (int, optional): Default batch size of the training data. Defaults to 32.
            - validation_batch_size (int, optional): Default batch size of the validation data. Defaults to 32.
            - test_batch_size (int, optional): Default batch size of the test data. Defaults to 32.
            - predict_batch_size (int, optional): Default batch size of the predict data. Defaults to 32.
            - train_transform (transforms.Compose, optional): Transformations to apply to the training dataset. Defaults to DEFAULT_TRANSFORMS.
            - val_transform (transforms.Compose, optional): Transformations to apply to the validation dataset. Defaults to DEFAULT_TRANSFORMS.
            - test_transform (transforms.Compose, optional): Transformations to apply to the testing dataset. Defaults to DEFAULT_TRANSFORMS.
            - predict_transform (transforms.Compose, optional): Transformations to apply to the prediction dataset. Defaults to DEFAULT_TRANSFORMS.
        Raises:
            - ValueError: The given dataset name is not supported. Supported: cifar10, cifar100.
        """
        super().__init__()
        self.data_dir: str = data_dir
        if dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(f"{dataset_name}: Not a supported dataset.")
        self.dataset_name: str = dataset_name
        self.train_transform: transforms.Compose = train_transforms
        self.val_transform: transforms.Compose = val_transforms
        self.test_transform: transforms.Compose = test_transforms
        self.predict_transform: transforms.Compose = predict_transforms
        self.validation_split: float = validation_split
        self.train_batch_size: int = train_batch_size
        self.validation_batch_size: int = validation_batch_size
        self.test_batch_size: int = test_batch_size
        self.predict_batch_size: int = predict_batch_size
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        """Downloading the data if not already available."""
        EMNIST(self.data_dir, train=True, split=self.dataset_name, download=True)
        EMNIST(self.data_dir, train=False, split=self.dataset_name, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup before training/testing/validation/prediction using the dataset.

        Args:
            - stage (Optional[str], optional): Current stage of the PyTorch training process used for setup. Defaults to None.
        """
        total_images: int = len(
            EMNIST(self.data_dir, train=True, split=self.dataset_name, download=True)
        )
        num_validation_images: int = int(total_images * self.validation_split)
        num_training_images: int = total_images - num_validation_images
        if (stage == "fit") or (not stage):
            self.emnist_train_full = EMNIST(
                self.data_dir,
                train=True,
                split=self.dataset_name,
                download=True,
                transform=self.train_transform,
            )
            if self.dataset_name == "letters":
                self.emnist_train_full.targets.apply_(
                    lambda x: (x - 1)
                )  # patch to eliminate the N/A label in the original letters dataset
            self.emnist_train, self.emnist_val = random_split(
                self.emnist_train_full, [num_training_images, num_validation_images]
            )

        if (stage == "test") or (not stage):
            self.emnist_test = EMNIST(
                self.data_dir,
                train=False,
                split=self.dataset_name,
                download=True,
                transform=self.test_transform,
            )
            if self.dataset_name == "letters":
                self.emnist_test.targets.apply_(
                    lambda x: (x - 1)
                )  # patch to eliminate the N/A label in the original letters dataset
        if (stage == "predict") or (not stage):
            self.emnist_predict = EMNIST(
                self.data_dir,
                train=False,
                split=self.dataset_name,
                download=True,
                transform=self.predict_transform,
            )
            if self.dataset_name == "letters":
                self.emnist_predict.targets.apply_(
                    lambda x: (x - 1)
                )  # patch to eliminate the N/A label in the original letters dataset

    def train_dataloader(self) -> DataLoader:
        """Training DataLoader wrapper.

        Returns:
            - DataLoader: PyTorch DataLoader object.
        """
        return DataLoader(
            self.emnist_train,
            batch_size=self.train_batch_size,
            num_workers=os.cpu_count() or 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation DataLoader wrapper.

        Returns:
            - DataLoader: PyTorch DataLoader object.
        """
        return DataLoader(
            self.emnist_val,
            batch_size=self.validation_batch_size,
            num_workers=os.cpu_count() or 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Test DataLoader wrapper.

        Returns:
            - DataLoader: PyTorch DataLoader object.
        """
        return DataLoader(
            self.emnist_test,
            batch_size=self.test_batch_size,
            num_workers=os.cpu_count() or 0,
        )

    def predict_dataloader(self) -> DataLoader:
        """Predict DataLoader object.

        Returns:
            - DataLoader: PyTorch DataLoader object.
        """
        return DataLoader(
            self.emnist_predict,
            batch_size=self.predict_batch_size,
            num_workers=os.cpu_count() or 0,
        )

    def federated_iid_dataloader(
        self, num_workers: int = 10, workers_batch_size: int = 10
    ) -> Dict[int, DataLoader]:
        """Loads the training dataset as iid split among the workers.

        Args:
            - num_workers (int, optional): number of workers for federated learning. Defaults to 10.
            - workers_batch_size (int, optional): batch size of the dataset for workers training locally. Defaults to 10.

        Returns:
            - Dict[int, DataLoader]: collection of workers as the keys and the PyTorch DataLoader object as values (used for training).
        """
        items: int = len(self.emnist_train_full) // num_workers
        distribution: np.ndarray = np.random.randint(
            low=0, high=len(self.emnist_train_full), size=(num_workers, items)
        )
        federated: Dict[int, DataLoader] = dict()
        for i in range(len(distribution)):
            federated[i] = DataLoader(
                DatasetSplit(self.emnist_train_full, distribution[i]),
                batch_size=workers_batch_size,
                shuffle=True,
                num_workers=os.cpu_count() or 0,
            )
        return federated

    def federated_non_iid_dataloader(
        self, num_workers: int = 10, workers_batch_size: int = 10, niid_factor: int = 2
    ) -> Dict[int, DataLoader]:
        """Loads the training dataset as non-iid split among the workers.

        Args:
            - num_workers (int, optional): number of workers for federated learning. Defaults to 10.
            - workers_batch_size (int, optional): batch size of the dataset for workers training locally. Defaults to 10.
            - niid_factor (int, optional): max number of classes held by each niid agent. lower the number, more measure of non-iidness. Defaults to 2.

        Returns:
            - Dict[int, DataLoader]: collection of workers as the keys and the PyTorch DataLoader object as values (used for training).
        """
        shards: int = num_workers * niid_factor
        items: int = len(self.emnist_train_full) // shards
        idx_shard: List[int] = list(range(shards))
        classes: np.ndarray = (
            np.array(self.emnist_train_full.targets)
            if isinstance(self.emnist_train_full.targets, list)
            else self.emnist_train_full.targets.numpy()
        )

        idxs_labels: np.ndarray = np.vstack(
            (np.arange(len(self.emnist_train_full)), classes)
        )
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs: np.ndarray = idxs_labels[0, :]
        distribution: Dict[int, np.ndarray] = {
            i: np.array([], dtype="int64") for i in range(num_workers)
        }
        np.random.seed(42)
        while idx_shard:
            for i in range(num_workers):
                rand_set: Set[int] = set(
                    np.random.choice(idx_shard, niid_factor, replace=False)
                )
                idx_shard = list(set(idx_shard) - rand_set)
                for rand in rand_set:
                    distribution[i] = np.concatenate(
                        (distribution[i], idxs[rand * items : (rand + 1) * items]),
                        axis=0,
                    )
        federated: Dict[int, DataLoader] = dict()
        for i in distribution:
            federated[i] = DataLoader(
                DatasetSplit(self.emnist_train_full, distribution[i]),
                batch_size=workers_batch_size,
                shuffle=True,
                num_workers=os.cpu_count() or 0,
            )
        return federated
