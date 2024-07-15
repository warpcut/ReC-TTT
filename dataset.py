from torchvision import transforms
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100
from torch.utils.data import random_split
import torch.multiprocessing
import torch
import torch.utils.data as data
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
from PIL import Image
import pickle
import os

def get_visda_transforms():
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    visda_train = transforms.Compose([transforms.Resize((256,256)),
                                  transforms.RandomCrop((224,224)),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean_train, std_train)])
    
    visda_test = transforms.Compose([transforms.Resize((256,256)),
                                transforms.CenterCrop((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean_train, std_train)])
    return visda_train, visda_test


def get_ttt_transforms():
    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip()
        ])
    return data_transforms

def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.4914, 0.4822, 0.4465] if mean_train is None else mean_train
    std_train = [0.247, 0.243, 0.261] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=mean_train,
                             std=std_train)
        ])
    return data_transforms

def get_data_transforms_test(size, isize, mean_train=None, std_train=None):
    mean_train = [0.4914, 0.4822, 0.4465] if mean_train is None else mean_train
    std_train = [0.247, 0.243, 0.261] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train,
                             std=std_train)
        ])

    return data_transforms

def CIFAR10_C(data_root, idx=0, level=1, data_transform=None, data_test_transform=None):
    train_data = CIFAR10(root=data_root, train=True, download=False, transform=data_transform)
    if idx == 99:
        test_data = CIFAR_New(root=data_root + '/CIFAR-10.1/', transform=data_test_transform)
        print(len(test_data))
        corruption_type = '10.1'
    elif idx == 100:
        corruption_type = 'None'
        test_data = CIFAR10(root=data_root,train=False, download=False, transform=data_test_transform)
    else:
        test_size = 10000
        corruption_types = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                            'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                            'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
        corruption_type = corruption_types[idx]
        teset_corr = np.load('{}/{}.npy'.format(data_root,corruption_type))
        teset_corr = teset_corr[(level-1)*test_size: level*test_size]
        test_data = CIFAR10(root=data_root,train=False, download=False, transform=data_test_transform)
        test_data.data = teset_corr
    return train_data, test_data, corruption_type

def CIFAR100_C(data_root, idx=0, level=1, data_transform=None, data_test_transform=None):
    train_data = CIFAR100(root=data_root, train=True, download=True, transform=data_transform)
    if idx == 100:
        corruption_type = 'None'
        test_data = CIFAR100(root=data_root, train=False, download=False, transform=data_test_transform)
    else:
        test_size = 10000
        corruption_types = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                            'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                            'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
        corruption_type = corruption_types[idx]
        teset_corr = np.load('{}/{}.npy'.format(data_root,corruption_type))
        teset_corr = teset_corr[(level-1)*test_size: level*test_size]
        test_data = CIFAR100(root=data_root,train=False, download=False, transform=data_test_transform)
        test_data.data = teset_corr
    return train_data, test_data, corruption_type

def VISDA(data_root, data_transform=None, data_test_transform=None, seed=111):
    train_data = ImageFolder(root=data_root + 'train/', transform=data_transform)
    train_data, val_split_data = random_split(train_data, [106678, 45719], generator=torch.Generator().manual_seed(seed))
    print('Training')
    val_data =ImageFolder(root=data_root + 'validation/', transform=data_test_transform)
    print('Val')
    test_data = VisdaTest(data_root, data_test_transform)
    return train_data, val_split_data, test_data, val_data

class VisdaTest(data.Dataset):
    def __init__(self, root, transforms = None):
        self.root = root
        self.transforms = transforms
        self.img_list = np.loadtxt(root + 'image_list.txt', dtype=str)

    def __len__(self):
        return self.img_list.shape[0]

    def __getitem__(self, idx):
        name = self.img_list[idx][0]
        label = int(self.img_list[idx][1])

        img = Image.open(self.root + 'test/' + name)
        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

class CIFAR_New(data.Dataset):
	def __init__(self, root, transform=None, target_transform=None, version='v6'):
		self.data = np.load('%s/cifar10.1_%s_data.npy' %(root, version))
		self.targets = np.load('%s/cifar10.1_%s_labels.npy' %(root, version)).astype('long')
		self.transform = transform
		self.target_transform = target_transform

	def __getitem__(self, index):
		img, target = self.data[index], self.targets[index]
		img = Image.fromarray(img)
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		return img, target

	def __len__(self):
		return len(self.targets)


class ImageNet32Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x.transpose((0, 2, 3, 1))
        self.y= y
        self.transform= transform
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        target = self.y[idx]
        img = Image.fromarray(self.x[idx])
        img = self.transform(img)
        return img,target


def ImageNet32(data_root):
    train_img_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
                            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_x, test_y = load_test(data_root)
    train_x,train_y = load_train(data_root)
    trainset = ImageNet32Dataset(train_x, train_y,transform=train_img_transform)
    valset = ImageNet32Dataset(test_x, test_y,transform=test_img_transform)
    return trainset, valset

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def load_train(data_folder='../../nas_data/imagenet/', img_size=32):
    def f(idx):
        data_file = os.path.join(data_folder, 'train_data_batch_')
        d = unpickle(data_file + str(idx))
        x = d['data']
        y = d['labels']
        y = [i-1 for i in y]
        data_size = x.shape[0]
        img_size2 = img_size * img_size
        x = np.dstack(
            (x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
        x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)
        return [x,y]
    data = [f(idx) for idx in range(1,11)]
    all_x =  np.concatenate([d[0] for d in data],axis=0)
    all_y = []
    for d in data:
        all_y.extend(d[1])

    return all_x,all_y

def load_test(data_folder='../../nas_data/imagenet/', img_size=32):
    data_file = os.path.join(data_folder, 'val_data')
    d = unpickle(data_file )
    x = d['data']
    y = d['labels']
    y = [i-1 for i in y]
    data_size = x.shape[0]
    img_size2 = img_size * img_size

    x = np.dstack(
        (x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    return x, y