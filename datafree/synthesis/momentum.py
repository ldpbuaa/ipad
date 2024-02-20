from .generative import GenerativeSynthesizer




class MomentumSynthesizer(GenerativeSynthesizer):
    def __init__(self, teacher, student, generator, nz, img_size, num_classes=100,
                iterations=1, lr_g=1e-3, synthesis_batch_size=128,
                sample_batch_size=128, adv=0, bn=0, oh=0, act=0, balance=0,
                rw=0, criterion=None, normalizer=None, device='cpu',
                 autocast=None, use_fp16=False, distributed=False):
        super(MomentumSynthesizer, self).__init__(
                teacher, student, generator, nz, img_size, num_classes,
                iterations, lr_g, synthesis_batch_size,
                sample_batch_size, adv, bn, oh, act, balance,
                rw, criterion, normalizer, device,
                 autocast, use_fp16, distributed)
