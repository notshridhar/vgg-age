import os
import numpy
import torch

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import datasets
from torchvision import transforms


# global variables
# can change from outside
random_scale = (0.4, 1.0)
mean = [0.5, 0.5, 0.5]
std = [0.2, 0.2, 0.2]


def get_transforms():
    global std
    global mean
    global random_scale

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop((224, 224), scale=random_scale),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return (train_transform, valid_transform)


def find_mean_std(train_dir):
    """
    Get the mean and std per channel
    very slow because of two passes

    parameters -------------------------
    - train_dir     -   path of training set

    returns ----------------------------
    - mean          -   mean of the dataset per channel
    - std           -   standard deviation per channel
    """

    pin_memory = True if torch.cuda.is_available() else False
    train_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    train_dataset = datasets.ImageFolder(train_dir, train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=pin_memory,
    )

    mn = torch.Tensor([0, 0, 0])
    st = torch.Tensor([0, 0, 0])
    count = len(train_loader)

    for input, target in train_loader:
        mn += input.mean([0, 2, 3])
    
    mn = mn / count

    for input, target in train_loader:
        ch0 = (input[0][0] - mn[0])
        ch1 = (input[0][1] - mn[1])
        ch2 = (input[0][2] - mn[2])
        st[0] += torch.mul(ch0, ch0).sum() / 50176
        st[1] += torch.mul(ch1, ch1).sum() / 50176
        st[2] += torch.mul(ch2, ch2).sum() / 50176

    # st = root(sum(x^2) / N)
    st = torch.sqrt(st / count)

    return (mn, st)


def split_loader(
    train_dir, valid_frac=0.1, batch_size=32, shuffle=True,
):
    """
    Function for splitting and loading train and valid iterators
    
    parameters -------------------------
    - train_dir     -   path of training set
    - valid_frac    -   fraction split of the training set used for validation
    - batch_size    -   how many samples per batch to load
    - shuffle       -   whether to shuffle the train or validation indices

    returns ----------------------------
    - train_loader  -   training set iterator
    - valid_loader  -   validation set iterator
    """

    # valid frac range assert
    error_msg = "Error : valid_frac should be in the range [0, 1]"
    assert (valid_frac >= 0) and (valid_frac <= 1), error_msg

    # override if cuda is available
    pin_memory = True if torch.cuda.is_available() else False

    # load as dataset
    train_transform, valid_transform = get_transforms()
    train_dataset = datasets.ImageFolder(train_dir, train_transform)
    valid_dataset = datasets.ImageFolder(train_dir, valid_transform)

    # get indices
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(valid_frac * num_train)

    # shuffle if required
    if shuffle:
        numpy.random.shuffle(indices)

    # samplers
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=0,
        pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)


def separate_loader(
    train_dir, valid_dir, batch_size=32, shuffle=True,
):
    """
    Function for splitting and loading train and valid iterators
    
    parameters -------------------------
    - train_dir     -   path of training set
    - valid_dir     -   path of validation set
    - batch_size    -   how many samples per batch to load
    - shuffle       -   whether to shuffle the train or validation indices

    returns ----------------------------
    - train_loader  -   training set iterator
    - valid_loader  -   validation set iterator
    """

    # load as dataset
    train_transform, valid_transform = get_transforms()
    train_dataset = datasets.ImageFolder(train_dir, train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, valid_transform)

    # override if cuda is available
    pin_memory = True if torch.cuda.is_available() else False

    # dataloaders
    train_loader = DataLoader(
        train_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=pin_memory,
    )

    valid_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)


def test_loader(test_dir, batch_size=32, shuffle=False):
    """
    Function for loading test image iterators
    
    parameters -------------------------
    - test_dir      -   path of image folder
    - batch_size    -   how many samples per batch to load

    returns ----------------------------
    - test_loader   -   data iterator
    """

    # override if cuda is available
    pin_memory = True if torch.cuda.is_available() else False

    # load as dataset
    _, valid_transform = get_transforms()
    test_dataset = datasets.ImageFolder(test_dir, valid_transform)

    # dataloaders
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )

    return test_loader


def load_pth(path):
    # just for code completeness
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.load(path, map_location=device)
