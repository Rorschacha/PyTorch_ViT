import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def random_plots(**kwargs):
    """
    随机展示一组图片(9个)数据
    :param kwargs: labels,path_anno
    :return:
    """
    key = kwargs.keys()
    if "labels" in key:
        labels = kwargs["labels"]
        if True:  # draw
            random_idx = np.random.randint(1, len(labels), size=9)
            fig, axes = plt.subplots(3, 3, figsize=(16, 12))

            for idx, ax in enumerate(axes.ravel()):
                img = mpimg.imread(labels[random_idx[idx]][0])
                ax.set_title(labels[random_idx[idx]][1])
                ax.imshow(img)
            plt.show()


def read_anno_pc(path_anno=r""):
    """读 图片分类 标注文件"""
    with open(path_anno, 'r') as fobj:
        txts = fobj.readlines()  # 是lines

    labels = []
    for anno in txts:  # "xxx.jpg    1/n"
        sr = anno.strip().split("    ")
        labels.append((sr[0], sr[1]))
    return labels


# Image Augumentation
train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

# Dataset
class CatsDogsDataset(Dataset):
    def __init__(self, labels, transform=None):
        self.file_list = labels
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx][0]
        #img = mpimg.imread(img_path)
        img = Image.open(img_path) #pillow 读取图片
        img_transformed = self.transform(img)

        label = int(self.file_list[idx][1])

        return img_transformed, label


def main():
    path_anno = r"D:\datasets\PracticeSets\cats and dogs\data\annotation_train.txt"
    labels = read_anno_pc(path_anno)
    # random_plots(labels=labels)

    # print("b")
    pass


if __name__ == '__main__':
    main()
