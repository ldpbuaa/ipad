import os
import torch
import torchvision
import datafree
import torch.nn as nn

from PIL import Image
from datafree.models import classifiers
from torchvision import datasets
from torchvision import transforms as T
from PIL import PngImagePlugin

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

NORMALIZE_DICT = {
    'mnist':    dict( mean=(0.1307,),                std=(0.3081,) ),
    'cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'cifar10-LT':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'cifar100': dict( mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761) ),
    'cifar100-LT': dict( mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761) ),
    'imagenet': dict( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'imagenet-LT': dict( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'tiny_imagenet': dict( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'tiny_imagenet-LT': dict( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'CUB-LT':   dict( mean=(0.451, 0.466, 0.414), std=(0.202, 0.204, 0.205) ),
    'food-LT':   dict( mean=(0.509, 0.414, 0.320), std=(0.256, 0.251, 0.237) ),
    'caltech256-LT':   dict( mean=(0.555, 0.534, 0.506), std=(0.220, 0.220, 0.223) ),
    'indoor-LT':   dict( mean=(0.484, 0.426, 0.368), std=(0.212, 0.208, 0.204) ),
    'places-LT':   dict( mean=(0, 0, 0), std=(1, 1, 1) ),
    'stanford_dogs':   dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'stanford_cars':   dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'places365_32x32': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'places365_64x64': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'places365': dict( mean=(0.458, 0.441, 0.408), std=(0.229, 0.226, 0.236) ),
    'svhn': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'imagenet_32x32': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),

    # for semantic segmentation
    'camvid': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'nyuv2': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
}


MODEL_DICT = {
    # https://github.com/polo5/ZeroShotKnowledgeTransfer
    'wrn16_1': classifiers.wresnet.wrn_16_1,
    'wrn16_2': classifiers.wresnet.wrn_16_2,
    'wrn40_1': classifiers.wresnet.wrn_40_1,
    'wrn40_2': classifiers.wresnet.wrn_40_2,

    # https://github.com/HobbitLong/RepDistiller
    'resnet8': classifiers.resnet_tiny.resnet8,
    'resnet20': classifiers.resnet_tiny.resnet20,
    'resnet32': classifiers.resnet_tiny.resnet32,
    'resnet56': classifiers.resnet_tiny.resnet56,
    'resnet110': classifiers.resnet_tiny.resnet110,
    'resnet8x4': classifiers.resnet_tiny.resnet8x4,
    'resnet32x4': classifiers.resnet_tiny.resnet32x4,

    # https://github.com/huawei-noah/Data-Efficient-Model-Compression/tree/master/DAFL
    'resnet50':  classifiers.resnet.resnet50,
    'resnet18':  classifiers.resnet.resnet18,
    'resnet34':  classifiers.resnet.resnet34,

    'vgg16': classifiers.vgg.vgg16,
}

IMAGENET_MODEL_DICT = {
    'resnet50': classifiers.resnet_in.resnet50,
    'resnet34': classifiers.resnet_in.resnet34,
    'resnet18': classifiers.resnet_in.resnet18,
    'mobilenetv2': torchvision.models.mobilenet_v2,
}


DATASET_INFO = {
    'cifar10':{
                'num_classes':10,
                'image_size':(3, 32, 32),
                'noise_dim': 256,
    },
    'cifar10-LT':{
                'num_classes':10,
                'image_size':(3, 32, 32),
                'noise_dim': 256,
    },
    'cifar100':{
                'num_classes':100,
                'image_size':(3, 32, 32),
                'noise_dim': 256,
    },
    'cifar100-LT':{
                'num_classes':100,
                'image_size':(3, 32, 32),
                'noise_dim': 256,
    },
    'imagenet':{
                'num_classes':1000,
                'image_size':(3, 224, 224),
                'noise_dim': 1024,
    },
    'imagenet-LT':{
                'num_classes':1000,
                'image_size':(3, 224, 224),
                'noise_dim': 1024,
    },
    'tiny_imagenet':{
                'num_classes':200,
                'image_size':(3, 64, 64),
                'noise_dim': 512,
    },
    'tiny_imagenet-LT':{
                'num_classes':200,
                'image_size':(3, 64, 64),
                'noise_dim': 512,
    },
    'food-LT':{
                'num_classes':101,
                'image_size':(3, 64, 64),
                'noise_dim': 512,
    },
    'CUB-LT':{
                'num_classes':200,
                'image_size':(3, 64, 64),
                'noise_dim': 512,
    },
    'caltech256-LT':{
                'num_classes':257,
                'image_size':(3, 64, 64),
                'noise_dim': 512,
    },
    'indoor-LT':{
                'num_classes':67,
                'image_size':(3, 64, 64),
                'noise_dim': 512,
    },
    'places':{
                'num_classes':365,
                'image_size':(3, 224, 224),
                'noise_dim': 1024,
    },
    'places-LT':{
                'num_classes':365,
                'image_size':(3, 224, 224),
                'noise_dim': 512,
    },
}

def get_model(dataset, name: str, num_classes, pretrained=False, **kwargs):
    if 'imagenet' in dataset or 'in' in name or 'places' in dataset:
    # if False:
        model = IMAGENET_MODEL_DICT[name](num_classes, pretrained=pretrained)
        if num_classes!=1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = MODEL_DICT[name](num_classes=num_classes)
    return model


def get_dataset(name: str, data_root: str='~/data', return_transform=False,
    split=['A', 'B', 'C', 'D'], imbalance_ratio=1.):
    data_root = os.path.expanduser( data_root )

    if name=='cifar10':
        num_classes = DATASET_INFO[name]['num_classes']
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'CIFAR10' )
        train_dst = datasets.CIFAR10(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.CIFAR10(data_root, train=False, download=True, transform=val_transform)
    elif name=='cifar10-LT':
        num_classes = DATASET_INFO[name]['num_classes']
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'CIFAR10' )
        train_dst = datafree.datasets.IMBALANCECIFAR10(data_root, train=True,
                        download=True, imbalance_ratio=imbalance_ratio,
                        transform=train_transform)
        val_dst = datafree.datasets.IMBALANCECIFAR10(data_root, train=False,
                        download=True, transform=val_transform)
    elif name=='cifar100':
        num_classes = DATASET_INFO[name]['num_classes']
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'CIFAR100' )
        train_dst = datasets.CIFAR100(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.CIFAR100(data_root, train=False, download=True, transform=val_transform)
    elif name=='cifar100-LT':
        num_classes = DATASET_INFO[name]['num_classes']
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'CIFAR100' )
        train_dst = datafree.datasets.IMBALANCECIFAR100(data_root, train=True,
                        download=True, imbalance_ratio=imbalance_ratio,
                        transform=train_transform)
        val_dst = datafree.datasets.IMBALANCECIFAR100(data_root, train=False,
                        download=True, transform=val_transform)
    elif name=='imagenet':
        num_classes = DATASET_INFO[name]['num_classes']
        train_transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join( data_root, 'imagenet' )
        train_dst = datasets.ImageNet(data_root, split='train', transform=train_transform)
        val_dst = datasets.ImageNet(data_root, split='val', transform=val_transform)
    elif name=='imagenet-LT':
        num_classes = DATASET_INFO[name]['num_classes']
        img_size = DATASET_INFO[name]['image_size'][1:]
        train_transform = T.Compose([
            # T.Resize(img_size),
            # T.RandomCrop(img_size[0], padding=8),
            T.RandomResizedCrop(img_size[0]),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join( data_root, 'imagenet')
        train_dst = datafree.datasets.ImageNet_LT(data_root, split='train',
                            imbalance_ratio=imbalance_ratio , transform=train_transform)
        val_dst = datafree.datasets.ImageNet_LT(data_root, split='val',
                            imbalance_ratio=imbalance_ratio, transform=val_transform)

        '''
        txt_path = 'datafree/datasets/ImageNet_LT/'
        train_dst = datafree.datasets.ImageNet_LT(data_root,
                            split_txt=f'{txt_path}/ImageNet_LT_train.txt',
                            transform=train_transform)
        val_dst = datafree.datasets.ImageNet_LT(data_root,
                            split_txt=f'{txt_path}/ImageNet_LT_val.txt',
                            transform=val_transform)
        '''
    elif name=='tiny_imagenet':
        num_classes = DATASET_INFO[name]['num_classes']
        train_transform = T.Compose([
            T.RandomCrop(64, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )
        data_root = os.path.join(data_root, 'tiny-imagenet-200')
        train_dst = datafree.datasets.TinyImageNet(data_root, split='train', transform=train_transform)
        val_dst = datafree.datasets.TinyImageNet(data_root, split='val', transform=val_transform)
    elif name=='tiny_imagenet-LT':
        num_classes = DATASET_INFO[name]['num_classes']
        train_transform = T.Compose([
            T.RandomCrop(64, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )
        data_root = os.path.join(data_root, 'tiny-imagenet-200')
        train_dst = datafree.datasets.TinyImageNet_LT(data_root, split='train',
                                                imbalance_ratio=imbalance_ratio,
                                                transform=train_transform)
        val_dst = datafree.datasets.TinyImageNet_LT(data_root, split='val', transform=val_transform)
    elif name=='CUB-LT':
        num_classes = DATASET_INFO[name]['num_classes']
        train_transform = T.Compose([
            T.Resize((64, 64)),
            T.RandomCrop(64, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )
        val_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )
        data_root = os.path.join(data_root, 'CUB_200_2011')
        train_dst = datafree.datasets.CUB_LT(data_root, split='train',
                                                imbalance_ratio=imbalance_ratio,
                                                transform=train_transform)
        val_dst = datafree.datasets.CUB_LT(data_root, split='test', transform=val_transform)
    elif name=='food-LT':
        num_classes = DATASET_INFO[name]['num_classes']
        train_transform = T.Compose([
            T.Resize((64, 64)),
            T.RandomCrop(64, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )
        val_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )
        data_root = os.path.join(data_root, 'Food101')
        train_dst = datafree.datasets.Food_LT(data_root, split='train',
                                                imbalance_ratio=imbalance_ratio,
                                                transform=train_transform)
        val_dst = datafree.datasets.Food_LT(data_root, split='test', transform=val_transform)
    elif name=='places-LT':
        num_classes = DATASET_INFO[name]['num_classes']
        img_size = DATASET_INFO[name]['image_size'][1:]
        train_transform = T.Compose([
            # T.Resize(img_size),
            # T.RandomCrop(img_size[0], padding=8),
            T.RandomResizedCrop(img_size[0]),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )
        val_transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )
        data_root = os.path.join(data_root, 'places365')
        train_dst = datafree.datasets.Places_LT(data_root, split='train',
                                                imbalance_ratio=imbalance_ratio,
                                                transform=train_transform)
        val_dst = datafree.datasets.Places_LT(data_root, split='val', transform=val_transform)
    elif name=='caltech256-LT':
        num_classes = DATASET_INFO[name]['num_classes']
        train_transform = T.Compose([
            T.Resize((64, 64)),
            T.RandomCrop(64, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )
        val_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )
        data_root = os.path.join(data_root, 'Caltech256')
        train_dst = datafree.datasets.Caltech256_LT(data_root, split='train',
                                                imbalance_ratio=imbalance_ratio,
                                                transform=train_transform)
        val_dst = datafree.datasets.Caltech256_LT(data_root, split='val', transform=val_transform)
    elif name=='indoor-LT':
        num_classes = DATASET_INFO[name]['num_classes']
        train_transform = T.Compose([
            T.Resize((64, 64)),
            T.RandomCrop(64, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )
        val_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )
        data_root = os.path.join(data_root, 'MIT_Indoors')
        train_dst = datafree.datasets.Indoor_LT(data_root, split='train',
                                                imbalance_ratio=imbalance_ratio,
                                                transform=train_transform)
        val_dst = datafree.datasets.Indoor_LT(data_root, split='test', transform=val_transform)
    else:
        raise NotImplementedError
    if return_transform:
        return num_classes, train_dst, val_dst, train_transform, val_transform
    return num_classes, train_dst, val_dst
