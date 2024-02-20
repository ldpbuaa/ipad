import torch
import copy
from torch.utils.data import ConcatDataset, Dataset
import numpy as np
from PIL import Image
import os, random, math
from copy import deepcopy
from contextlib import contextmanager
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import cm
import io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('INFO')
from ..criterions import KLDiv

def get_pseudo_label(n_or_label, num_classes, device, onehot=False):
    if isinstance(n_or_label, int):
        label = torch.randint(0, num_classes, size=(n_or_label,), device=device)
    else:
        label = n_or_label.to(device)
    if onehot:
        label = torch.zeros(len(label), num_classes, device=device).scatter_(1, label.unsqueeze(1), 1.)
    return label

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    r"""Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)

class MemoryBank(object):
    def __init__(self, device, max_size=4096, dim_feat=512):
        self.device = device
        self.data = torch.randn( max_size, dim_feat ).to(device)
        self._ptr = 0
        self.n_updates = 0

        self.max_size = max_size
        self.dim_feat = dim_feat

    def add(self, feat):
        n, c = feat.shape
        assert self.dim_feat==c and self.max_size % n==0, "%d, %d"%(self.dim_feat, c, self.max_size, n)
        self.data[self._ptr:self._ptr+n] = feat.detach()
        self._ptr = (self._ptr+n) % (self.max_size)
        self.n_updates+=n

    def get_data(self, k=None, index=None):
        if k is None:
            k = self.max_size
        assert k <= self.max_size

        if self.n_updates>self.max_size:
            if index is None:
                index = random.sample(list(range(self.max_size)), k=k)
            return self.data[index], index
        else:
            if index is None:
                index = random.sample(list(range(self._ptr)), k=min(k, self._ptr))
            return self.data[index], index

def clip_images(image_tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor

def save_image_batch(imgs, output, col=None, size=None, pred=None, pack=True):
    if isinstance(imgs, torch.Tensor):
        imgs = (imgs.detach().clamp(0, 1).cpu().numpy()*255).astype('uint8')
    base_dir = os.path.dirname(output)
    if base_dir!='':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        imgs = pack_images( imgs, col=col ).transpose( 1, 2, 0 ).squeeze()
        imgs = Image.fromarray( imgs )
        if size is not None:
            if isinstance(size, (list,tuple)):
                imgs = imgs.resize(size)
            else:
                w, h = imgs.size
                max_side = max( h, w )
                scale = float(size) / float(max_side)
                _w, _h = int(w*scale), int(h*scale)
                imgs = imgs.resize([_w, _h])
        imgs.save(output)
    else:
        output_filename = output.strip('.png')
        for idx, img in enumerate(imgs):
            img = Image.fromarray( img.transpose(1, 2, 0) )
            if pred is not None:
                save_name = f'{output_filename}-{idx}-{pred[idx]}.png'
            else:
                save_name = f'{output_filename}-{idx}.png'
            img.save(save_name)

def pack_images(images, col=None, channel_last=False, padding=1):
    # N, C, H, W
    if isinstance(images, (list, tuple) ):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0,3,1,2) # make it channel first
    assert len(images.shape)==4
    assert isinstance(images, np.ndarray)

    N,C,H,W = images.shape
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))

    pack = np.zeros( (C, H*row+padding*(row-1), W*col+padding*(col-1)), dtype=images.dtype )
    for idx, img in enumerate(images):
        h = (idx // col) * (H+padding)
        w = (idx % col) * (W+padding)
        pack[:, h:h+H, w:w+W] = img
    return pack

def flatten_dict(dic):
    flattned = dict()
    def _flatten(prefix, d):
        for k, v in d.items():
            if isinstance(v, dict):
                if prefix is None:
                    _flatten( k, v )
                else:
                    _flatten( prefix+'/%s'%k, v )
            else:
                if prefix is None:
                    flattned[k] = v
                else:
                    flattned[ prefix+'/%s'%k ] = v

    _flatten(None, dic)
    return flattned

def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [ -m / s for m, s in zip(mean, std) ]
        _std = [ 1/s for s in std ]
    else:
        _mean = mean
        _std = std

    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor

class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, reverse=False):
        return normalize(x, self.mean, self.std, reverse=reverse)

def load_yaml(filepath):
    yaml=YAML()
    with open(filepath, 'r') as f:
        return yaml.load(f)

def _collect_all_images(root, postfix=['png', 'jpg', 'jpeg', 'JPEG']):
    images = []
    if isinstance( postfix, str):
        postfix = [ postfix ]
    for dirpath, dirnames, files in os.walk(root):
        for pos in postfix:
            for f in files:
                if f.endswith( pos ):
                    images.append( os.path.join( dirpath, f ) )
    return images

class UnlabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = os.path.abspath(root)
        self.images = _collect_all_images(self.root) #[ os.path.join(self.root, f) for f in os.listdir( root ) ]
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open( self.images[idx] )
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return 'Unlabeled data:\n\troot: %s\n\tdata mount: %d\n\ttransforms: %s'%(self.root, len(self), self.transform)

class LabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = os.path.abspath(root)
        self.categories = [ int(f) for f in os.listdir( root ) ]
        images = []
        targets = []
        for c in self.categories:
            category_dir = os.path.join( self.root, str(c))
            _images = [ os.path.join( category_dir, f ) for f in os.listdir(category_dir) ]
            images.extend(_images)
            targets.extend([c for _ in range(len(_images))])
        self.images = images
        self.targets = targets
        self.transform = transform
    def __getitem__(self, idx):
        img, target = Image.open( self.images[idx] ), self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, target
    def __len__(self):
        return len(self.images)

class ImagePool(object):
    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._idx = 0

    def add(self, imgs, targets=None):
        save_image_batch(imgs, os.path.join( self.root, "%d.png"%(self._idx) ), pack=False)
        self._idx+=1

    def get_dataset(self, transform=None, labeled=True):
        return UnlabeledImageDataset(self.root, transform=transform)

class DataIter(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)

    def next(self):
        try:
            data = next( self._iter )
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next( self._iter )
        return data

@contextmanager
def dummy_ctx(*args, **kwds):
    try:
        yield None
    finally:
        pass

def freeze_backbone(model):
    for n, p in model.named_parameters():
        if ('fc' not in n) and ('linear' not in n) and ('classifier' not in n):
            p.requires_grad = False
    model.eval() # fix BN and Dropout
    return model


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)

def get_shot_thld(class_count):
    """ get many / low shot threshold
    """
    sorted_class_count = copy.deepcopy(class_count)
    sorted_class_count.sort(reverse=True)
    num_per_cls = len(sorted_class_count) // 3
    return sorted_class_count[num_per_cls], sorted_class_count[-(num_per_cls+1)]

def plot_shot_acc(shot_accs_his):
    shot_accs = {'many':[], 'medium':[], 'few':[]}
    for accs in shot_accs_his:
        shot_accs['many'].append(accs[0])
        shot_accs['medium'].append(accs[1])
        shot_accs['few'].append(accs[2])
    for k, v in shot_accs.items():
        plt.plot(np.array(v), label=k)
    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.savefig('./shot_accs_plot.png')
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close()
    img_buf.seek(0)
    tf_img = tf.image.decode_png(img_buf.getvalue(), channels=4)
    tf_img = tf.expand_dims(tf_img, 0)
    return tf_img

def plot_shot_argmax_dist(shot_argmax_his, train_class_counts, name=None):
    shot_dist = {'many':[], 'medium':[], 'few':[]}
    for shot_argmax in shot_argmax_his:
        for k,v in shot_argmax.items():
            shot_dist[k].append(v)
    for k, v in shot_dist.items():
        plt.plot(np.array(v), label=k)
    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('Counts')
    plt.savefig(f'./{name}_shot_argmax_dist_plot.png')
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close()
    img_buf.seek(0)
    tf_img = tf.image.decode_png(img_buf.getvalue(), channels=4)
    tf_img = tf.expand_dims(tf_img, 0)
    return tf_img


def plot_shot_kl(shot_kl_his):
    shot_kl = {'many':[], 'medium':[], 'few':[]}
    for kl_loss in shot_kl_his:
        shot_kl['many'].append(kl_loss['many'])
        shot_kl['medium'].append(kl_loss['medium'])
        shot_kl['few'].append(kl_loss['few'])
    for k, v in shot_kl.items():
        plt.plot(np.array(v), label=k)
    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('KL divergence')
    plt.savefig('./shot_kl_plot.png')
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close()
    img_buf.seek(0)
    tf_img = tf.image.decode_png(img_buf.getvalue(), channels=4)
    tf_img = tf.expand_dims(tf_img, 0)
    return tf_img


def plot_class_acc(class_accs_log, class_counts):
    """ visuallize class acc history
    """
    cls_accs_log = copy.deepcopy(class_accs_log)
    if not torch.is_tensor(cls_accs_log):
        cls_accs_log = torch.vstack(cls_accs_log)
    if len(cls_accs_log.shape) == 1:
        cls_accs_log = cls_accs_log.unsqueeze(0)
    x = np.arange(cls_accs_log.shape[0])
    y = cls_accs_log.numpy() # accs
    length, num_class = np.shape(y)
    color = cm.rainbow(np.linspace(0, 1, num_class))
    # acc_log plot
    plt.figure(figsize=(48,48))
    for i in range(num_class):
        plt.plot(x, y[:, i], c=color[i], label=f'class: {i+1} #:{int(class_counts[i])}')
    plt.title(f'Class Acc History')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.grid()
    plt.savefig('./class_accs_plot.png')
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close()
    img_buf.seek(0)
    tf_img = tf.image.decode_png(img_buf.getvalue(), channels=4)
    tf_img = tf.expand_dims(tf_img, 0)
    return tf_img


def pred_matrix(pred_mat, t_out, s_out):
    t_pred = torch.argmax(t_out.clone().detach().cpu(), dim=1)
    s_pred = torch.argmax(s_out.clone().detach().cpu(), dim=1)
    for t, s in zip(t_pred, s_pred):
        pred_mat[t, s] += 1
    return pred_mat

def plot_pred_matrix(pred_mat, epoch, save_dir='./pred_mat'):
    os.makedirs(save_dir, exist_ok=True)
    sns.heatmap(pred_mat.numpy())
    plt.xlabel('student prediction')
    plt.ylabel('teacher prediction')
    plt.show()
    plt.savefig(f'{save_dir}/pred_matrix_epoch_{epoch}.png')
    plt.close()

def shot_kld_loss(t_preds, s_preds, train_class_counts):
    """ KL loss for multi-shots
    """
    shot_kld = {'many':0, 'medium':0, 'few':0}
    KLD = KLDiv()
    t_preds, s_preds = torch.vstack(t_preds), torch.vstack(s_preds)
    many_shot_thr, low_shot_thr = get_shot_thld(train_class_counts)
    for i in range(len(train_class_counts)):
        if train_class_counts[i] > many_shot_thr:
            indices = torch.argmax(s_preds, dim=1) == torch.tensor(i)
            if indices.sum() > 0:
                shot_kld['many'] += KLD(s_preds[indices], t_preds[indices])
        elif train_class_counts[i] < low_shot_thr:
            indices = torch.argmax(s_preds, dim=1) == torch.tensor(i)
            if indices.sum() > 0:
                shot_kld['few'] += KLD(s_preds[indices], t_preds[indices])
        else:
            indices = torch.argmax(s_preds, dim=1) == torch.tensor(i)
            if indices.sum() > 0:
                shot_kld['medium'] += KLD(s_preds[indices], t_preds[indices])
    return shot_kld


def get_shot_num(train_class_counts):
    shot_num = {'many':0, 'medium':0, 'few':0}
    many_shot_thr, low_shot_thr = get_shot_thld(train_class_counts)
    for i in train_class_counts:
        if i > many_shot_thr:
            shot_num['many'] += 1
        elif i < low_shot_thr:
            shot_num['few'] += 1
        else:
            shot_num['medium'] += 1
    return shot_num


def shot_argmax_dist(argmax_dist, train_class_counts):
    """ shot argmax normalized by the number of class of each shot
    """
    shot_num = get_shot_num(train_class_counts)
    argmax_dist = torch.cat(argmax_dist)
    shot_argmax = {'many':0, 'medium':0, 'few':0}
    many_shot_thr, low_shot_thr = get_shot_thld(train_class_counts)
    for i in range(len(train_class_counts)):
        if train_class_counts[i] > many_shot_thr:
            shot_argmax['many'] +=  (argmax_dist == torch.tensor(i)).sum() / shot_num['many']
        elif train_class_counts[i] < low_shot_thr:
            shot_argmax['few'] +=  (argmax_dist == torch.tensor(i)).sum() / shot_num['few']
        else:
            shot_argmax['medium'] +=  (argmax_dist == torch.tensor(i)).sum() / shot_num['medium']
    return shot_argmax


def getStat(train_data):
    """Get the mean and std value for a certain dataset."""
    print('Compute mean and variance for training data.')
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=4,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    print(f'mean:{mean} and std:{std}')