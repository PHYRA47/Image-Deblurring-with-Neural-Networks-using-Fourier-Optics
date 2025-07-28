from torch.utils.data import DataLoader
from .dataset import ImageDataset
import torchvision.transforms as T
import parameters as params


def get_dataloader(
    train_dir:      str,
    test_dir:       str,
    psf_bank_path:  str,
    train_batch_size:   int,
    test_batch_size:    int,
    shuffle: bool = True,
    verbose: bool = False,
    **kwargs
):
    batch_sizes = {
        "train": train_batch_size,
        "test": test_batch_size
    }

    transforms = {
        "train": T.Compose([
            T.ToTensor(),
        ]),
        "test": T.Compose([
            T.ToTensor(),
        ])
    }
    
    image_datasets = {
        "train": ImageDataset(
            image_dir=train_dir,
            psf_bank_path=psf_bank_path,
            transform=transforms["train"],
            verbose=verbose,
        ),
        "test": ImageDataset(
            image_dir=test_dir,
            psf_bank_path=psf_bank_path,
            transform=transforms["test"],
            verbose=verbose
        )
    }

    dataloaders = {
        split: DataLoader(
            dataset=image_datasets[split],
            batch_size=batch_sizes[split],
            shuffle=(split == "train" and shuffle),
            **kwargs
        ) for split in ["train", "test"]
    }

    if verbose:
        print("DataLoader created.")
        print(f"  Train directory       : {train_dir}")
        print(f"  Test directory        : {test_dir}")
        print(f"  PSF bank path         : {psf_bank_path}")
        print(f"  Train samples         : {len(image_datasets['train'])}")
        print(f"  Test samples          : {len(image_datasets['test'])}")
        print(f"  Train batch size      : {train_batch_size}")
        print(f"  Test batch size       : {test_batch_size}")
        print(f"  Shuffle train         : {shuffle}")

    return dataloaders

if __name__ == "__main__":
    dataloaders = get_dataloader(
        train_dir="dataset/train",
        test_dir="dataset/test",
        psf_bank_path="dataset/psf_bank.npz",
        train_batch_size=1,
        test_batch_size=1,
        shuffle=True,
        verbose=True
    )
