import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import datafree
import time

from .base import BaseSynthesis
from datafree.hooks import DeepInversionHook
from datafree.criterions import kldiv, get_image_prior_losses
from datafree.utils import ImagePool, DataIter, clip_images
from datafree.utils.loss import DecorrLoss

class GenerativeSynthesizer(BaseSynthesis):
    def __init__(self, teacher, student, generator, nz, img_size, train_class_counts,
                 num_classes=100, iterations=1, lr_g=1e-3, synthesis_batch_size=128,
                sample_batch_size=128, adv=0, bn=0, oh=0, act=0, balance=0,
                rw=0, decorr=0, criterion=None, normalizer=None, device='cpu',
                 autocast=None, scaler=None, use_fp16=False, distributed=False, momentum=0.95,
                 ):
        super(GenerativeSynthesizer, self).__init__(teacher, student)
        assert len(img_size)==3, "image size should be a 3-dimension tuple"
        self.img_size = img_size
        self.num_classes = num_classes
        self.iterations = iterations
        self.nz = nz
        if criterion is None:
            criterion = kldiv
        self.criterion = criterion
        self.normalizer = normalizer
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size

        # scaling factors
        self.lr_g = lr_g
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.balance = balance
        self.act = act
        self.decorr = decorr

        # generator
        self.generator = generator.to(device).train()
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(0.9,0.999))
        self.distributed = distributed
        self.use_fp16 = use_fp16
        self.autocast = autocast # for FP16
        self.scaler = scaler
        self.device = device

        # hooks for deepinversion regularization
        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append( DeepInversionHook(m) )

        # class adative entropy regularization
        self.rw = rw
        self.reweight = (rw > 0)
        self.reweight_lr = 0.1
        self.epsilon = 0.1

        self.momentum = momentum
        self.train_class_counts = train_class_counts
        self.epoch = -1

    def reset(self, ):
        self.class_weights = torch.ones((self.num_classes,)).to(self.device)
        self.cls_kl = torch.zeros((self.num_classes), ).to(self.device)
        self.cls_pred = torch.zeros((self.num_classes), ).to(self.device)
        self.kl_loss_mean = torch.zeros((self.num_classes), ).to(self.device)
        self.rw_loss_mean = torch.zeros((self.num_classes), ).to(self.device)

    def synthesize(self, epoch, step):
        self.epoch = epoch
        if step == 0:
            self.reset()
        start = time.time()
        if self.epoch % 10 == 0 and step == 0:
            self.reset()
        self.student.eval()
        self.generator.train()
        self.teacher.eval()
        t_preds, s_preds = [], []
        loss_bn_total, loss_oh_total, loss_adv_total, loss_act_total, loss_balance_total, loss_decorr_total = 0, 0, 0, 0, 0, 0
        for it in range(self.iterations):
            self.optimizer.zero_grad()
            with self.autocast(enabled=self.use_fp16):
                z = torch.randn( size=(self.synthesis_batch_size, self.nz), device=self.device)
                inputs = self.generator(z)
                inputs = self.normalizer(inputs)
                t_out, t_feat = self.teacher(inputs, return_features=True)
                s_out , s_feat = self.student(inputs, return_features=True)
            loss_bn = sum([h.r_feature for h in self.hooks]) # BN mean and var regularization
            loss_oh = F.cross_entropy( t_out, t_out.max(1)[1] ) # one-hot loss
            loss_act = - t_feat.abs().mean() # activation maximization

            if self.adv>0:
                if self.reweight and self.epoch > 2:
                    self.cls_kl_mmt(t_out, s_out)
                    self.cls_weights_opt()
                    loss_adv = self.cls_reweight_loss(s_out, t_out) # adversarial learning with loss reweighting
                else:
                    loss_adv = -self.criterion(s_out, t_out).sum() / s_out.size(0) # adversarial learning
            else:
                loss_adv = loss_oh.new_zeros(1)[0]
            p = F.softmax(t_out, dim=1).mean(0)
            loss_balance = (p * torch.log(p)).sum() # maximization entropy of predictions
            loss_decorr = DecorrLoss(s_feat)

            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv + \
                    self.balance * loss_balance + self.act * loss_act + self.decorr * loss_decorr
            if self.use_fp16:
                scaler = self.scaler
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            loss_bn_total += loss_bn.detach().cpu()
            loss_oh_total += loss_oh.detach().cpu()
            loss_adv_total += loss_adv.detach().cpu()
            loss_act_total += loss_act.detach().cpu()
            loss_balance_total += loss_balance.detach().cpu()
            loss_decorr_total += loss_decorr.detach().cpu()
            t_preds.append(t_out.detach().cpu())
            s_preds.append(s_out.detach().cpu())
        end = time.time()
        return { 'synthetic': self.normalizer(inputs.detach(), reverse=True),
                'time': int(end - start),
                'loss_bn_total':loss_bn_total,
                'loss_oh_total':loss_oh_total,
                'loss_adv_total':loss_adv_total,
                'loss_act_total':loss_act_total,
                'loss_balance_total':loss_balance_total,
                'loss_decorr_total':loss_decorr_total,
                }

    @torch.no_grad()
    def sample(self):
        self.generator.eval()
        z = torch.randn( size=(self.sample_batch_size, self.nz), device=self.device )
        inputs = self.normalizer(self.generator(z))
        return inputs

    def cls_kl_mmt(self, t_out, s_out):
        kl_div = -self.criterion(s_out, t_out).detach()
        preds = torch.argmax(t_out, dim=1)
        for i in torch.arange(0, self.num_classes-1, dtype=torch.long):
            indices = (preds==i)
            self.cls_kl[i] += 0.9 * kl_div[indices].sum()

    def cls_weights_opt(self, ):
        self.class_weights += self.reweight_lr * (
                            self.cls_kl - torch.mean(self.cls_kl) - self.epsilon)
        self.class_weights = torch.clamp(self.class_weights, min=0, max=200)

    def cls_reweight_loss(self, s_out, t_out):
        loss = -self.criterion(s_out, t_out).sum(dim=1)
        preds = torch.argmax(t_out, dim=1)
        weights = (self.class_weights[preds] - self.class_weights.mean())
        weights /= weights.norm(p=1)
        rw_loss = loss * (1 + self.rw * weights)
        return rw_loss.sum() / s_out.size(0) # convert to batchmean

    def loss_mean(self, loss, rw_loss, preds, it):
        for i in torch.arange(0, self.num_classes-1, dtype=torch.long):
            idx = (preds==i)
            self.kl_loss_mean[i] += 0.9 * loss[idx].sum()
            self.rw_loss_mean[i] += 0.9 * rw_loss[idx].sum()

    def kl_loss_epoch(self, loss, preds):
        cls_kl = torch.zeros((self.num_classes), ).to(self.device)
        for i in torch.arange(0, self.num_classes-1, dtype=torch.long):
            idx = (preds==i)
            cls_kl[i] = loss[idx].sum()
        self.kl_loss_mean = torch.vstack((self.kl_loss_mean, cls_kl))

    def cls_preds_epoch(self, preds):
        for i in torch.arange(0, self.num_classes-1, dtype=torch.long):
            idx = (preds==i)
            self.cls_pred[i] += idx.sum()
        self.cls_pred = torch.zeros((self.num_classes), ).to(self.device)

    def momentum_update(self, fast_states):
        slow_states = self.generator.state_dict()
        for k, v in fast_states.items():
            slow_states[k] = self.momentum * slow_states[k] + (1-self.momentum) * v
        self.generator.load_state_dict(slow_states)
