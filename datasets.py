
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import torch
from PIL import Image

def build_transform(args,is_train=True):
    if args.image_size==224:
        src_size=256
    elif args.image_size==448:
        src_size=480

    if is_train:
        transform = transforms.Compose([
            transforms.Resize(src_size),
            transforms.RandomCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(src_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    return transform

def build_dataloader(args):
    transform_train = build_transform(args, is_train=True)
    dataset_train=ImageFolder(os.path.join(args.dataset_path,"train"),
                              transform_train)
    dataloader_train=DataLoader(dataset_train,batch_size=args.batch_size,
                                shuffle=True,num_workers=args.nthreads,drop_last=True)

    transform_val = build_transform(args, is_train=False)
    dataset_val = ImageFolder(os.path.join(args.dataset_path, "val"),
                              transform_val)
    dataloader_val = DataLoader(dataset_val, batch_size=8,
                                  shuffle=False, num_workers=args.nthreads)

    num_classes=len(dataset_train.classes)
    n_iter_per_epoch_train=len(dataloader_train)

    return dataloader_train,dataloader_val,dataset_train,dataset_val,\
           num_classes,n_iter_per_epoch_train

