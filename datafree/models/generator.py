import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return torch.flatten(x, 1)

class Generator(nn.Module):
    """ nz: dimension of latent vector z
        nl: dimension of label embedding input vector
    """
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3, nl=100, num_classes=100):
        super(Generator, self).__init__()
        self.gan_type = 'condgan' if nl>0 else 'gan'
        self.num_classes = num_classes

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))
        if self.gan_type == 'condgan':
            self.emb = torch.nn.Embedding(num_classes, nl)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z, labels=None):
        self.device = z.device
        if self.gan_type=='condgan':
            labels = labels or self.gen_labels(z.shape[0])
            z += self.emb(labels)
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    def gen_labels(self, num=1):
        labels = torch.randint(low=0, high=self.num_classes-1, size=(num,))
        labels = F.one_hot(labels, self.num_classes).float().to(self.device)
        return labels


class LargeGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3, nl=100, num_classes=100):
        super(LargeGenerator, self).__init__()
        self.gan_type = 'condgan' if nl>0 else 'gan'
        self.num_classes = num_classes

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 4 * self.init_size ** 2)) #(512, 64 * 4 * 16 * 16)
        if self.gan_type == 'condgan':
            self.emb = torch.nn.Embedding(num_classes, nl)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 4), # 256
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*4, ngf*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z, labels=None):
        self.device = z.device
        if self.gan_type=='condgan':
            labels = labels or self.gen_labels(z.shape[0])
            z += self.emb(labels)
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    def gen_labels(self, num=1):
        labels = torch.randint(low=0, high=self.num_classes-1, size=(num,))
        labels = F.one_hot(labels, num_classes=self.num_classes).float()
        labels = torch.argmax(labels, dim=1).to(device=self.device)
        return labels

'''
class HugeGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3, nl=100, num_classes=100):
        super(HugeGenerator, self).__init__()
        self.gan_type = 'condgan' if nl>0 else 'gan'
        self.num_classes = num_classes

        self.init_size = img_size // 32
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * self.init_size ** 2))
        if self.gan_type == 'condgan':
            self.emb = torch.nn.Embedding(num_classes, nl)

        self.conv_blocks = nn.Sequential(
            BasicBlock(ngf, ngf), # 512 512
            nn.Upsample(scale_factor=2),
            BasicBlock(ngf, ngf//2), # 512 256
            nn.Upsample(scale_factor=2),
            BasicBlock(ngf//2, ngf//4), # 256 128
            nn.Upsample(scale_factor=2),
            BasicBlock(ngf//4, ngf//8), # 128 64
            nn.Upsample(scale_factor=2),
            BasicBlock(ngf//8, ngf//16), # 64, 32
            nn.Upsample(scale_factor=2),
            BasicBlock(ngf//16, ngf//32), # 32, 16
            nn.Conv2d(ngf//32, 3, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, z, labels=None):
        self.device = z.device
        if self.gan_type=='condgan':
            labels = labels or self.gen_labels(z.shape[0])
            z += self.emb(labels)
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        out = self.conv_blocks(out)
        img = torch.sigmoid(out)
        return img

    def gen_labels(self, num=1):
        labels = torch.randint(low=0, high=self.num_classes-1, size=(num,))
        labels = F.one_hot(labels, num_classes=self.num_classes).float()
        labels = torch.argmax(labels, dim=1).to(device=self.device)
        return labels
'''


class HugeGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3, nl=100, num_classes=100):
        super(HugeGenerator, self).__init__()
        self.gan_type = 'condgan' if nl>0 else 'gan'
        self.num_classes = num_classes

        self.init_size = img_size // 32
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 8 * self.init_size ** 2)) #(1024, 64 * 28 * 28)
        if self.gan_type == 'condgan':
            self.emb = torch.nn.Embedding(num_classes, nl)

        self.conv_blocks = nn.Sequential(
            BasicBlock(ngf*8, ngf*8), # 512 512
            nn.Upsample(scale_factor=2),
            BasicBlock(ngf*8, ngf*4), # 512 256
            nn.Upsample(scale_factor=2),
            BasicBlock(ngf*4, ngf*2), # 256 128
            nn.Upsample(scale_factor=2),
            BasicBlock(ngf*2, ngf), # 128 64
            nn.Upsample(scale_factor=2),
            BasicBlock(ngf, ngf//2), # 64 32
            nn.Upsample(scale_factor=2),
            BasicBlock(ngf//2, ngf//4), # 32 16
            nn.Conv2d(ngf//4, 3, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, z, labels=None):
        self.device = z.device
        if self.gan_type=='condgan':
            labels = labels or self.gen_labels(z.shape[0])
            z += self.emb(labels)
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        out = self.conv_blocks(out)
        img = torch.sigmoid(out)
        # print(f'debug image shape:{img.shape}')
        return img

    def gen_labels(self, num=1):
        labels = torch.randint(low=0, high=self.num_classes-1, size=(num,))
        labels = F.one_hot(labels, num_classes=self.num_classes).float()
        labels = torch.argmax(labels, dim=1).to(device=self.device)
        return labels





class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.LReLU1 =  nn.LeakyReLU(0.2)
        self.LReLU1 =  nn.ReLU()
        self.bn1 = nn.BatchNorm2d(planes)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.LReLU2 =  nn.LeakyReLU(0.2)
        #self.LReLU2 =  nn.ReLU()
        #self.bn2 = nn.BatchNorm2d(planes)

        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion*planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(self.expansion*planes)
        #     )

    def forward(self, x):
        # out = self.bn1(self.LReLU1(self.conv1(x)))
        # out = self.bn2(self.LReLU2(self.conv2(out)))
        out = self.LReLU1(self.bn1(self.conv1(x)))
        # out = self.LReLU2(self.bn2(self.conv2(out)))
        #out += self.shortcut(x)
        #out = F.relu(out)
        # print(out.shape)
        return out
