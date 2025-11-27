import os
import numpy as np
import torch
import random
from torchvision import datasets, transforms
from torchvision.transforms import functional as FUNC
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split, Subset

class Cutout(object):
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length
    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img


class RandAugment:
    def __init__(self, n=2, m=9):
        self.n = n
        self.m = m
        self.transforms = [
            self.auto_contrast,
            self.equalize,
            self.rotate,
            self.solarize,
            self.color,
            self.posterize,
            self.contrast,
            self.brightness,
            self.sharpness,
            self.shear_x,
            self.shear_y,
            self.translate_x,
            self.translate_y,
        ]
    def __call__(self, img):
        ops = random.choices(self.transforms, k=self.n)
        for op in ops:
            magnitude = self.m
            img = op(img, magnitude)
        return img
    def auto_contrast(self, img, magnitude):
        return FUNC.autocontrast(img)
    def equalize(self, img, magnitude):
        return FUNC.equalize(img)
    def rotate(self, img, magnitude):
        angle = magnitude * 3.0
        return FUNC.rotate(img, angle)
    def solarize(self, img, magnitude):
        threshold = 256 - (magnitude * 25.6)
        return FUNC.solarize(img, threshold)
    def color(self, img, magnitude):
        factor = 1.0 + magnitude * 0.09
        return FUNC.adjust_saturation(img, factor)
    def posterize(self, img, magnitude):
        bits = 8 - (magnitude * 0.7)
        return FUNC.posterize(img, int(bits))
    def contrast(self, img, magnitude):
        factor = 1.0 + magnitude * 0.09
        return FUNC.adjust_contrast(img, factor)
    def brightness(self, img, magnitude):
        factor = 1.0 + magnitude * 0.09
        return FUNC.adjust_brightness(img, factor)
    def sharpness(self, img, magnitude):
        factor = 1.0 + magnitude * 0.09
        return FUNC.adjust_sharpness(img, factor)
    def shear_x(self, img, magnitude):
        factor = magnitude * 0.03
        return FUNC.affine(img, angle=0, translate=(0, 0), scale=1.0, shear=(factor, 0))
    def shear_y(self, img, magnitude):
        factor = magnitude * 0.03
        return FUNC.affine(img, angle=0, translate=(0, 0), scale=1.0, shear=(0, factor))
    def translate_x(self, img, magnitude):
        pixels = magnitude * 3
        return FUNC.affine(img, angle=0, translate=(pixels, 0), scale=1.0, shear=(0, 0))
    def translate_y(self, img, magnitude):
        pixels = magnitude * 3
        return FUNC.affine(img, angle=0, translate=(0, pixels), scale=1.0, shear=(0, 0))

def mixup_collate_fn(batch, alpha=1.0):
    images = torch.stack([item[0] for item in batch], 0)
    labels = torch.tensor([item[1] for item in batch])
    batch_size = len(batch)
    indices = torch.randperm(batch_size)
    shuffled_images = images[indices]
    shuffled_labels = labels[indices]
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1




    mixed_images = lam * images + (1 - lam) * shuffled_images
    return mixed_images, labels, shuffled_labels, lam


class Ali_DataLoader:
    def __init__(
        self,
        data_dir="data",
        batch_size=32,
        random_seed=42,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        classes=None,
        train_transforms=None,
        test_transforms=None,
        augmentation_type="standard",
        mixup_alpha=1.0,
        cutout_params=None,
        randaugment_params=None,
    ):
        if cutout_params is None:
            cutout_params = {"n_holes": 1, "length": 16}
        if randaugment_params is None:
            randaugment_params = {"n": 2, "m": 9}
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.classes = classes
        self.augmentation_type = augmentation_type
        self.mixup_alpha = mixup_alpha
        self.cutout_params = cutout_params
        self.randaugment_params = randaugment_params
        self.use_mixup = augmentation_type == "mixup"
        try:
            os.makedirs(data_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory {data_dir}: {e}")
            raise
        self.cifar_mean = (0.4914, 0.4822, 0.4465)
        self.cifar_std = (0.2023, 0.1994, 0.2010)
        if train_transforms is None:
            if self.augmentation_type == "none":
                self.train_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),]
                )
            elif self.augmentation_type == "standard":
                self.train_transform = transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.2
                        ),
                        transforms.ToTensor(),
                    ]
                )
            elif self.augmentation_type == "cutout":
                self.train_transform = transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.2
                        ),
                        transforms.ToTensor(),
                        Cutout(
                            n_holes=self.cutout_params["n_holes"],
                            length=self.cutout_params["length"],
                        ),
                        transforms.Normalize(self.cifar_mean, self.cifar_std),
                    ]
                )
            elif self.augmentation_type == "randaugment":
                self.train_transform = transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        RandAugment(
                            n=self.randaugment_params["n"],
                            m=self.randaugment_params["m"],
                        ),
                        transforms.ToTensor(),
                        transforms.Normalize(self.cifar_mean, self.cifar_std),
                    ]
                )
            elif self.augmentation_type == "advanced":
                self.train_transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
                        transforms.RandomApply(
                            [
                                transforms.RandomRotation(15),
                            ],
                            p=0.5,
                        ),
                        transforms.RandomApply(
                            [
                                transforms.RandomAffine(
                                    0,
                                    translate=(0.1, 0.1),
                                    scale=(0.8, 1.2),
                                    shear=10,
                                )
                            ],
                            p=0.5,
                        ),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply(
                            [
                                transforms.ColorJitter(
                                    brightness=0.2,
                                    contrast=0.2,
                                    saturation=0.2,
                                    hue=0.1,
                                )
                            ],
                            p=0.8,
                        ),
                        transforms.RandomGrayscale(p=0.1),
                        transforms.ToTensor(),
                        Cutout(n_holes=2, length=8),
                        transforms.Normalize(self.cifar_mean, self.cifar_std),
                    ]
                )
            elif self.augmentation_type == "mixup":
                self.train_transform = transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]
                )
            else:
                raise ValueError(f"Unknown augmentation type: {self.augmentation_type}")
        else:
            self.train_transform = train_transforms

        if test_transforms is None:
            self.test_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        else:
            self.test_transform = test_transforms

        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def _get_subset_indices(self, dataset, classes):
        if classes is None:
            return list(range(len(dataset)))

        indices = []
        if isinstance(dataset, Subset):
            for i, idx in enumerate(dataset.indices):
                if dataset.dataset.targets[idx] in classes:
                    indices.append(i)
        else:
            for i, label in enumerate(dataset.targets):
                if label in classes:
                    indices.append(i)
        return indices

    def load_data(
        self,
        train_percent=0.7,
        val_percent=0.15,
        test_percent=0.15,
        subset_percent=100.0,
    ):
        assert (
            abs(train_percent + val_percent + test_percent - 1.0) < 1e-5
        ), "Split percentages must sum to 1"

        try:
            full_dataset = datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                download=True,
                transform=self.train_transform,
            )

            test_dataset = datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                download=True,
                transform=self.test_transform,
            )
        except Exception as e:
            print(f"Error loading CIFAR-10 dataset: {e}")
            raise

        if subset_percent < 1.0:
            print(f"Using only {subset_percent:.1%} of the total dataset")

            train_size = len(full_dataset)
            subset_size = int(train_size * subset_percent)
            indices = torch.randperm(train_size)[:subset_size].tolist()
            full_dataset = Subset(full_dataset, indices)

            test_size = len(test_dataset)
            test_subset_size = int(test_size * subset_percent)
            test_indices = torch.randperm(test_size)[:test_subset_size].tolist()
            test_dataset = Subset(test_dataset, test_indices)

        if self.classes is not None:
            train_indices = [
                i
                for i, label in enumerate(full_dataset.targets)
                if label in self.classes
            ]
            filtered_train_dataset = Subset(full_dataset, train_indices)

            test_indices = [
                i
                for i, label in enumerate(test_dataset.targets)
                if label in self.classes
            ]
            filtered_test_dataset = Subset(test_dataset, test_indices)

            working_train_dataset = filtered_train_dataset
            working_test_dataset = filtered_test_dataset
        else:
            working_train_dataset = full_dataset
            working_test_dataset = test_dataset

        dataset_size = len(working_train_dataset)
        train_size = int(train_percent / (train_percent + val_percent) * dataset_size)
        val_size = dataset_size - train_size

        train_subset, val_subset = random_split(
            working_train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.random_seed),
        )

        val_dataset_no_aug = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=False,
            transform=self.test_transform,
        )

        if self.classes is not None:
            val_indices = val_subset.indices
            filtered_val_dataset = Subset(val_dataset_no_aug, val_indices)
            val_subset_no_aug = filtered_val_dataset
        else:
            val_subset_no_aug = Subset(val_dataset_no_aug, val_subset.indices)

        if self.augmentation_type == "mixup":
            mixup_collate = lambda batch: mixup_collate_fn(batch, alpha=self.mixup_alpha)
            self.train_loader = DataLoader(
                train_subset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=mixup_collate,
            )
            self.use_mixup = True
            print(f"Using Mixup augmentation with alpha={self.mixup_alpha}")
        else:
            self.train_loader = DataLoader(
                train_subset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            self.use_mixup = False
        self.val_loader = DataLoader(
            val_subset_no_aug,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.test_loader = DataLoader(
            working_test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.train_dataset = train_subset
        self.val_dataset = val_subset
        self.test_dataset = working_test_dataset
        print(
            f"Data split into {train_percent:.0%} train, {val_percent:.0%} validation, {test_percent:.0%} test"
        )
        print(f"Dataset stored in: {os.path.abspath(self.data_dir)}")
        if self.classes is not None:
            class_names = [self.get_class_names()[c] for c in self.classes]
            print(f"Using classes: {class_names}")

        if self.augmentation_type not in ["none", "mixup"]:
            print(f"Using {self.augmentation_type} augmentation for training data")
        return self.train_loader, self.val_loader, self.test_loader
    def get_class_names(self):
        return [
            "airplane",
            "automobile",
            "bird",
            "cat",

            "deer", "dog",
            "frog",
            "horse","ship", "truck",]
    def get_class_distribution(self, loader):
        class_names = self.get_class_names()
        class_counts = {i: 0 for i in range(len(class_names))}
        dataset = loader.dataset
        if isinstance(dataset, Subset):
            for idx in dataset.indices:
                label = dataset.dataset.targets[idx]
                if label in class_counts:
                    class_counts[label] += 1
        else:
            for label in dataset.targets:
                if label in class_counts:
                    class_counts[label] += 1
        named_counts = {
            class_names[i]: count for i, count in class_counts.items() if count > 0
        }
        return named_counts
