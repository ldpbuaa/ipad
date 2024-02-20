from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import os

def loader(path):
    return Image.open(path)

class ImageNet_LT(ImageFolder):
    cls_num = 1000
    def __init__(self, root = '~/data', split='train', imbalance_ratio=1., imb_type='exp', transform=None):
        root_path = os.path.join(root, split)
        super().__init__(root_path, transform=transform)
        self.train = split == 'train'
        if self.train:
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imbalance_ratio)
            self.get_class_images()
            self.gen_imbalanced_data(img_num_list)
        self.transform = transform
        self.labels = self.targets
        phase = 'Train' if self.train else 'Evaluation'
        print(f"{phase} dataset: Contain {len(self.samples)} images with imbalanced_ratio: {imbalance_ratio}")

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.samples) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def get_class_images(self, ):
        self.class_images = { c:[] for c in range(self.cls_num)}
        for s in self.samples:
            self.class_images[int(s[1])].append(s)

    def gen_imbalanced_data(self, img_num_per_cls):
        # sort class according to num of images and change the img_num_per_cls
        self.class_image_nums = [len(self.class_images[c]) for c in range(self.cls_num)]
        self.class_sorted_index = np.argsort(-np.array(self.class_image_nums)) #decending
        sorted_img_num_per_cls = np.zeros(self.cls_num, dtype=np.uint32)
        for i in range(self.cls_num):
            sorted_img_num_per_cls[self.class_sorted_index[i]] = img_num_per_cls[i]
        samples, targets = [], []
        classes = np.arange(self.cls_num)
        self.num_per_cls_dict = dict()
        for c, img_num in zip(classes, sorted_img_num_per_cls):
            self.num_per_cls_dict[c] = img_num
            samples.extend(self.class_images[c][:img_num])
            targets.extend([c,] * img_num )
        self.imgs = self.samples = samples
        self.targets = targets

    def __getitem__(self, index):
        path = self.samples[index][0]
        label = self.samples[index][1]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.targets)
