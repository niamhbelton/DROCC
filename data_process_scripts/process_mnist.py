'''
Code borrowed from https://github.com/lukasruff/Deep-SVDD-PyTorch
'''
from PIL import Image
import numpy as np
from random import sample
from abc import ABC, abstractmethod
import torch
from torch.utils.data import Subset
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random

class BaseADDataset(ABC):
    """Anomaly detection dataset base class."""

    def __init__(self, root: str):
        super().__init__()
        self.root = root  # root path to data

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = None  # tuple with original class labels that define the normal class
        self.outlier_classes = None  # tuple with original class labels that define the outlier class

        self.train_set = None  # must be of type torch.utils.data.Dataset
        self.test_set = None  # must be of type torch.utils.data.Dataset
        self.task = None
        self.N = None

    @abstractmethod
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        """Implement data loaders of type torch.utils.data.DataLoader for train_set and test_set."""
        pass

    def __repr__(self):
        return self.__class__.__name__

class TorchvisionDataset(BaseADDataset):
    """TorchvisionDataset class for datasets already implemented in torchvision.datasets."""

    def __init__(self, root: str):
        super().__init__(root)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)


        return train_loader, test_loader

class MNIST_Dataset(TorchvisionDataset):

    def __init__(self, root: str, contam, N, normal_class=0, task='train'):
        super().__init__(root)
        #Loads only the digit 0 and digit 1 data
        # for both train and test
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)
        self.N = N
        self.contam = contam

        transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.1307],
                                                std=[0.3081])])

        target_transform = transforms.Lambda(lambda x: int(x not in self.outlier_classes))

        train_set = MyMNIST(root=self.root, train=True, download=True,
                              transform=transform, target_transform=target_transform)

        # Subset train set to normal class
        train_idx_normal = get_target_label_idx(train_set.targets, self.normal_classes, self.contam, task)


        if N > 0:
          np.random.seed(1)
          ind = random.sample(range(0, len(train_idx_normal)), self.N)
          train_idx_normal=np.array(train_idx_normal)[ind]

        self.train_set = Subset(train_set, train_idx_normal)

        if task == 'test':
          self.test_set = MyMNIST(root=self.root, train=False, download=True,
                                    transform=transform, target_transform=target_transform)
        else:
          test_set = MyMNIST(root=self.root, train=True, download=True,
                                    transform=transform, target_transform=target_transform)
          test_idx_normal = get_target_label_idx(train_set.targets, self.normal_classes, 'val')
          self.test_set = Subset(test_set, test_idx_normal)



class MyMNIST(MNIST):
    """Torchvision MNIST class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MyMNIST, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed


def get_target_label_idx(labels, targets, contam, task):
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    """
    if task == 'train':
      norm = np.argwhere(np.isin(labels, targets)).flatten().tolist()
      np.random.seed(50)
      ind = random.sample(range(0, len(norm)), len(norm) - 150)
      return np.array(norm)[ind]
    elif task == 'val':
      norm = np.argwhere(np.isin(labels, targets)).flatten().tolist()
      not_norm = np.argwhere(~np.isin(labels, targets)).flatten().tolist()

      #get the indexes that are being used for training
      np.random.seed(50)
      ind = random.sample(range(0, len(norm)), len(norm) - 150)

      #get indexes that are not used for training
      lst = list(range(0,len(norm) ))
      ind2 = [x for i,x in enumerate(lst) if i not in ind] #normals for validation

      ind3 = random.sample(range(0, len(not_norm)), 1500 - 150) #anoms
      anoms = np.array(not_norm)[ind3]

      norms = np.array(norm)[ind2]
      return anoms.tolist() + norms.tolist()
    else:
        if contam > 0.0:

            ind = np.where(np.array(labels)==targets[0])[0] #get indexes in the training set that are equal to the normal class
            poll = np.ceil(len(ind) * contam)
            random.seed(1)
            samp = random.sample(range(0, len(ind)), len(ind) - int(poll)) #randomly sample len(ind) - poll normal data points
            final_indexes = ind[samp]
            con = np.where(np.array(labels)!=targets[0])[0] #get indexes of non-normal class
            samp2 = random.sample(range(0, len(con)), int(poll))
            return np.array(list(final_indexes) + list(con[samp2]))
        else:
            return np.argwhere(np.isin(labels, targets)).flatten().tolist()




def global_contrast_normalization(x: torch.tensor, scale='l2'):
    """
    Apply global contrast normalization to tensor, i.e. subtract mean across features (pixels) and normalize by scale,
    which is either the standard deviation, L1- or L2-norm across features (pixels).
    Note this is a *per sample* normalization globally across features (and not across the dataset).
    """

    assert scale in ('l1', 'l2')

    n_features = int(np.prod(x.shape))

    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean

    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))

    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features

    x /= x_scale

    return x
