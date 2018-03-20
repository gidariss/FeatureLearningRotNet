from __future__ import print_function
import torch
import torch.utils.data as data
import torchvision
import torchnet as tnt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from Places205 import Places205
import numpy as np
import random
from torch.utils.data.dataloader import default_collate
from PIL import Image
import os
import errno
import numpy as np
import sys
import csv

from pdb import set_trace as breakpoint

# Set the paths of the datasets here.
_CIFAR_DATASET_DIR = './datasets/CIFAR'
_IMAGENET_DATASET_DIR = './datasets/IMAGENET/ILSVRC2012'
_PLACES205_DATASET_DIR = './datasets/Places205'


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds

class Places205(data.Dataset):
    def __init__(self, root, split, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.data_folder  = os.path.join(self.root, 'data', 'vision', 'torralba', 'deeplearning', 'images256')
        self.split_folder = os.path.join(self.root, 'trainvalsplit_places205')
        assert(split=='train' or split=='val')
        split_csv_file = os.path.join(self.split_folder, split+'_places205.csv')

        self.transform = transform
        self.target_transform = target_transform
        with open(split_csv_file, 'rb') as f:
            reader = csv.reader(f, delimiter=' ')
            self.img_files = []
            self.labels = []
            for row in reader:
                self.img_files.append(row[0])
                self.labels.append(long(row[1]))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        image_path = os.path.join(self.data_folder, self.img_files[index])
        img = Image.open(image_path).convert('RGB')
        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.labels)

class GenericDataset(data.Dataset):
    def __init__(self, dataset_name, split, random_sized_crop=False,
                 num_imgs_per_cat=None):
        self.split = split.lower()
        self.dataset_name =  dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
        self.random_sized_crop = random_sized_crop

        # The num_imgs_per_cats input argument specifies the number
        # of training examples per category that would be used.
        # This input argument was introduced in order to be able
        # to use less annotated examples than what are available
        # in a semi-superivsed experiment. By default all the 
        # available training examplers per category are being used.
        self.num_imgs_per_cat = num_imgs_per_cat

        if self.dataset_name=='imagenet':
            assert(self.split=='train' or self.split=='val')
            self.mean_pix = [0.485, 0.456, 0.406]
            self.std_pix = [0.229, 0.224, 0.225]

            if self.split!='train':
                transforms_list = [
                    transforms.Scale(256),
                    transforms.CenterCrop(224),
                    lambda x: np.asarray(x),
                ]
            else:
                if self.random_sized_crop:
                    transforms_list = [
                        transforms.RandomSizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        lambda x: np.asarray(x),
                    ]
                else:
                    transforms_list = [
                        transforms.Scale(256),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        lambda x: np.asarray(x),
                    ]
            self.transform = transforms.Compose(transforms_list)
            split_data_dir = _IMAGENET_DATASET_DIR + '/' + self.split
            self.data = datasets.ImageFolder(split_data_dir, self.transform)
        elif self.dataset_name=='places205':
            self.mean_pix = [0.485, 0.456, 0.406]
            self.std_pix = [0.229, 0.224, 0.225]
            if self.split!='train':
                transforms_list = [
                    transforms.CenterCrop(224),
                    lambda x: np.asarray(x),
                ]
            else:
                if self.random_sized_crop:
                    transforms_list = [
                        transforms.RandomSizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        lambda x: np.asarray(x),
                    ]
                else:
                    transforms_list = [
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        lambda x: np.asarray(x),
                    ]
            self.transform = transforms.Compose(transforms_list)
            self.data = Places205(root=_PLACES205_DATASET_DIR, split=self.split,
                transform=self.transform)
        elif self.dataset_name=='cifar10':
            self.mean_pix = [x/255.0 for x in [125.3, 123.0, 113.9]]
            self.std_pix = [x/255.0 for x in [63.0, 62.1, 66.7]]

            if self.random_sized_crop:
                raise ValueError('The random size crop option is not supported for the CIFAR dataset')

            transform = []
            if (split != 'test'):
                transform.append(transforms.RandomCrop(32, padding=4))
                transform.append(transforms.RandomHorizontalFlip())
            transform.append(lambda x: np.asarray(x))
            self.transform = transforms.Compose(transform)
            self.data = datasets.__dict__[self.dataset_name.upper()](
                _CIFAR_DATASET_DIR, train=self.split=='train',
                download=True, transform=self.transform)
        else:
            raise ValueError('Not recognized dataset {0}'.format(dname))
        
        if num_imgs_per_cat is not None:
            self._keep_first_k_examples_per_category(num_imgs_per_cat)
        
    
    def _keep_first_k_examples_per_category(self, num_imgs_per_cat):
        print('num_imgs_per_category {0}'.format(num_imgs_per_cat))
   
        if self.dataset_name=='cifar10':
            labels = self.data.test_labels if (self.split=='test') else self.data.train_labels
            data = self.data.test_data if (self.split=='test') else self.data.train_data
            label2ind = buildLabelIndex(labels)
            all_indices = []
            for cat in label2ind.keys():
                label2ind[cat] = label2ind[cat][:num_imgs_per_cat]
                all_indices += label2ind[cat]
            all_indices = sorted(all_indices)
            data = data[all_indices]
            labels = [labels[idx] for idx in all_indices]
            if self.split=='test':
                self.data.test_labels = labels
                self.data.test_data = data
            else: 
                self.data.train_labels = labels
                self.data.train_data = data

            label2ind = buildLabelIndex(labels)
            for k, v in label2ind.items(): 
                assert(len(v)==num_imgs_per_cat)

        elif self.dataset_name=='imagenet':
            raise ValueError('Keeping k examples per category has not been implemented for the {0}'.format(dname))
        elif self.dataset_name=='place205':
            raise ValueError('Keeping k examples per category has not been implemented for the {0}'.format(dname))
        else:
            raise ValueError('Not recognized dataset {0}'.format(dname))


    def __getitem__(self, index):
        img, label = self.data[index]
        return img, int(label)

    def __len__(self):
        return len(self.data)

class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def rotate_img(img, rot):
    if rot == 0: # 0 degrees rotation
        return img
    elif rot == 90: # 90 degrees rotation
        return np.flipud(np.transpose(img, (1,0,2)))
    elif rot == 180: # 90 degrees rotation
        return np.fliplr(np.flipud(img))
    elif rot == 270: # 270 degrees rotation / or -90
        return np.transpose(np.flipud(img), (1,0,2))
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


class DataLoader(object):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 unsupervised=True,
                 epoch_size=None,
                 num_workers=0,
                 shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.batch_size = batch_size
        self.unsupervised = unsupervised
        self.num_workers = num_workers

        mean_pix  = self.dataset.mean_pix
        std_pix   = self.dataset.std_pix
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_pix, std=std_pix)
        ])
        self.inv_transform = transforms.Compose([
            Denormalize(mean_pix, std_pix),
            lambda x: x.numpy() * 255.0,
            lambda x: x.transpose(1,2,0).astype(np.uint8),
        ])

    def get_iterator(self, epoch=0):
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)
        if self.unsupervised:
            # if in unsupervised mode define a loader function that given the
            # index of an image it returns the 4 rotated copies of the image
            # plus the label of the rotation, i.e., 0 for 0 degrees rotation,
            # 1 for 90 degrees, 2 for 180 degrees, and 3 for 270 degrees.
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img0, _ = self.dataset[idx]
                rotated_imgs = [
                    self.transform(img0),
                    self.transform(rotate_img(img0,  90)),
                    self.transform(rotate_img(img0, 180)),
                    self.transform(rotate_img(img0, 270))
                ]
                rotation_labels = torch.LongTensor([0, 1, 2, 3])
                return torch.stack(rotated_imgs, dim=0), rotation_labels
            def _collate_fun(batch):
                batch = default_collate(batch)
                assert(len(batch)==2)
                batch_size, rotations, channels, height, width = batch[0].size()
                batch[0] = batch[0].view([batch_size*rotations, channels, height, width])
                batch[1] = batch[1].view([batch_size*rotations])
                return batch
        else: # supervised mode
            # if in supervised mode define a loader function that given the
            # index of an image it returns the image and its categorical label
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img, categorical_label = self.dataset[idx]
                img = self.transform(img)
                return img, categorical_label
            _collate_fun = default_collate

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
            load=_load_function)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
            collate_fn=_collate_fun, num_workers=self.num_workers,
            shuffle=self.shuffle)
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size / self.batch_size

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    dataset = GenericDataset('imagenet','train', random_sized_crop=True)
    dataloader = DataLoader(dataset, batch_size=8, unsupervised=True)

    for b in dataloader(0):
        data, label = b
        break

    inv_transform = dataloader.inv_transform
    for i in range(data.size(0)):
        plt.subplot(data.size(0)/4,4,i+1)
        fig=plt.imshow(inv_transform(data[i]))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.show()
