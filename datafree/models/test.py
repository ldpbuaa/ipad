from generator import HugeGenerator
import torch
device = 'cuda:0'
z = torch.randn( size=(1, 1000), device=device)
generator = HugeGenerator(nz=1000, ngf=64, img_size=224, num_classes=1000, nl=1000).to(device)

img = generator(z)
print(img.shape)
