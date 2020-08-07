import os
import torch.utils.data as data
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
import random


class dataset_single(data.Dataset):

    def __init__(self, opts, setname, input_dim):
        self.dataroot = opts.dataroot
        images = os.listdir(os.path.join(self.dataroot, opts.phase + setname))
        self.img = [os.path.join(self.dataroot, opts.phase + setname, x) for x in images]
        self.size = len(self.img)
        self.input_dim = input_dim

        # setup image transformation
        transforms = [Resize(opts.resize_size, Image.BICUBIC)]
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)
        print('%s: %d images' % (setname, self.size))
        return

    def __getitem__(self, index):
        data = self.load_img(self.img[index], self.input_dim)
        name_img = os.path.basename(self.img[index]).split('.')[0]
        return name_img, data

    def load_img(self, img_name, input_dim):

        img = Image.open(img_name).convert('RGB')
        img = self.transforms(img)
        if input_dim == 1:
            img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = img.unsqueeze(0)
        return img

    def __len__(self):
        return self.size


class dataset_single_multi(data.Dataset):

    def __init__(self, opts, setname, input_dim):
        self.dataroot = opts.dataroot
        self.data_flair = os.path.join(self.dataroot, 'Flair')
        self.data_t1 = os.path.join(self.dataroot, 'T1')
        # images = os.listdir(os.path.join(self.dataroot, opts.phase + setname))
        images_flair = sorted(os.listdir(os.path.join(self.data_flair, opts.phase + setname)))
        images_t1 = sorted(os.listdir(os.path.join(self.data_t1, opts.phase + setname)))
        self.img_flair = [os.path.join(self.data_flair, opts.phase + setname, x) for x in images_flair]
        self.img_t1 = [os.path.join(self.data_t1, opts.phase + setname, x) for x in images_t1]

        self.size = len(self.img_flair)
        self.input_dim = input_dim

        # setup image transformation
        transforms = [Resize(opts.resize_size, Image.BICUBIC)]
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)
        print('%s: %d images' % (setname, self.size))
        return

    def __getitem__(self, index):

        name_img = os.path.basename(self.img_flair[index]).split('.')[0]
        data = torch.cat(
            (self.load_img(self.img_flair[index], self.input_dim), self.load_img(self.img_t1[index], self.input_dim)),
            dim=0)
        return name_img, data

    def load_img(self, img_name, input_dim):
        img = Image.open(img_name).convert('RGB')
        img = self.transforms(img)
        if input_dim == 1:
            img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = img.unsqueeze(0)
        return img

    def __len__(self):
        return self.size


class dataset_unpair(data.Dataset):
    def __init__(self, opts):
        self.dataroot = opts.dataroot

        # A
        images_A = os.listdir(os.path.join(self.dataroot, opts.phase + 'A'))
        self.A = [os.path.join(self.dataroot, opts.phase + 'A', x) for x in images_A]

        # B
        images_B = os.listdir(os.path.join(self.dataroot, opts.phase + 'B'))
        self.B = [os.path.join(self.dataroot, opts.phase + 'B', x) for x in images_B]

        self.A_size = len(self.A)
        self.B_size = len(self.B)
        self.dataset_size = max(self.A_size, self.B_size)
        self.input_dim_A = opts.input_dim_a
        self.input_dim_B = opts.input_dim_b

        # setup image transformation
        transforms = [Resize(opts.resize_size, Image.BICUBIC)]
        if opts.phase == 'train':
            transforms.append(RandomCrop(opts.crop_size))
        else:
            transforms.append(CenterCrop(opts.crop_size))
        if not opts.no_flip:
            transforms.append(RandomHorizontalFlip())
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)
        print('A: %d, B: %d images' % (self.A_size, self.B_size))
        return

    def __getitem__(self, index):
        if self.dataset_size == self.A_size:
            data_A = self.load_img(self.A[index], self.input_dim_A)
            data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)], self.input_dim_B)
        else:
            data_A = self.load_img(self.A[random.randint(0, self.A_size - 1)], self.input_dim_A)
            data_B = self.load_img(self.B[index], self.input_dim_B)
        return data_A, data_B

    def load_img(self, img_name, input_dim):
        img = Image.open(img_name).convert('RGB')
        img = self.transforms(img)
        if input_dim == 1:
            img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = img.unsqueeze(0)
        return img

    def __len__(self):
        return self.dataset_size


class dataset_unpair_multi(data.Dataset):
    def __init__(self, opts):
        self.dataroot = opts.dataroot
        self.data_flair = os.path.join(self.dataroot, 'Flair')
        self.data_t1 = os.path.join(self.dataroot, 'T1')

        # A
        images_A_flair = sorted(os.listdir(os.path.join(self.data_flair, opts.phase + 'A')))
        images_A_t1 = sorted(os.listdir(os.path.join(self.data_t1, opts.phase + 'A')))
        self.A_flair = [os.path.join(self.data_flair, opts.phase + 'A', x) for x in images_A_flair]
        self.A_t1 = [os.path.join(self.data_t1, opts.phase + 'A', x) for x in images_A_t1]

        # B
        images_B_flair = sorted(os.listdir(os.path.join(self.data_flair, opts.phase + 'B')))
        images_B_t1 = sorted(os.listdir(os.path.join(self.data_t1, opts.phase + 'B')))
        self.B_flair = [os.path.join(self.data_flair, opts.phase + 'B', x) for x in images_B_flair]
        self.B_t1 = [os.path.join(self.data_t1, opts.phase + 'B', x) for x in images_B_t1]

        self.A_size = len(self.A_flair)
        self.B_size = len(self.B_flair)
        self.dataset_size = max(self.A_size, self.B_size)
        self.input_dim_A = opts.input_dim_a
        self.input_dim_B = opts.input_dim_b

        # setup image transformation
        transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
        if opts.phase == 'train':
            transforms.append(RandomCrop(opts.crop_size))
        else:
            transforms.append(CenterCrop(opts.crop_size))
        if not opts.no_flip:
            transforms.append(RandomHorizontalFlip())
        transforms.append(ToTensor())
        transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transforms = Compose(transforms)
        print('A: %d, B: %d images' % (self.A_size, self.B_size))
        return

    def __getitem__(self, index):
        if self.dataset_size == self.A_size:
            data_A = torch.cat((self.load_img(self.A_flair[index], self.input_dim_A),
                                self.load_img(self.A_t1[index], self.input_dim_A)), dim=0)
            ran_no = random.randint(0, self.B_size - 1)
            data_B = torch.cat((self.load_img(self.B_flair[ran_no], self.input_dim_B),
                                self.load_img(self.B_t1[ran_no], self.input_dim_B)), dim=0)
            # print(data_A.size())
            # print(self.B_flair[ran_no], self.B_t1[ran_no])
        else:
            ran_no = random.randint(0, self.A_size - 1)
            data_A = torch.cat((self.load_img(self.A_flair[ran_no], self.input_dim_A),
                                self.load_img(self.A_t1[ran_no], self.input_dim_A)), dim=0)
            data_B = torch.cat((self.load_img(self.B_flair[index], self.input_dim_B),
                                self.load_img(self.B_t1[index], self.input_dim_B)), dim=0)
        return data_A, data_B

    def load_img(self, img_name, input_dim):
        img = Image.open(img_name).convert('RGB')
        img = self.transforms(img)
        if input_dim == 1:
            img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = img.unsqueeze(0)
        return img

    def __len__(self):
        return self.dataset_size
