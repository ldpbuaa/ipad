from tqdm import tqdm
import torch.nn.functional as F
import torch
import numpy as np
import torchmetrics
from . import metrics
from datafree.utils import get_shot_thld
from datafree.utils.misc import get_shot_num

class Evaluator(object):
    def __init__(self, metric, dataloader):
        self.dataloader = dataloader
        self.metric = metric

    def eval(self, model, device=None, progress=False):
        self.metric.reset()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate( tqdm(self.dataloader, disable=not progress) ):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model( inputs )
                self.metric.update(outputs, targets)
        return self.metric.get_results()

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)


class AdvEvaluator(object):
    def __init__(self, metric, dataloader, adversary):
        self.dataloader = dataloader
        self.metric = metric
        self.adversary = adversary

    def eval(self, model, device=None, progress=False):
        self.metric.reset()
        for i, (inputs, targets) in enumerate( tqdm(self.dataloader, disable=not progress) ):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = self.adversary.perturb(inputs, targets)
            with torch.no_grad():
                outputs = model( inputs )
                self.metric.update(outputs, targets)
        return self.metric.get_results()

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)


class DDP_Evaluator(object):
    def __init__(self, metric, dataloader):
        self.dataloader = dataloader
        self.metric = metric

    def eval(self, model, device=None, progress=False):
        self.metric.reset()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate( tqdm(self.dataloader, disable=not progress) ):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model( inputs )
                self.metric(outputs, targets)
        acc = self.metric.compute()
        loss = 0.
        return {'Acc':(acc.detach().cpu(), 0.), 'Loss':loss}

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)


def classification_evaluator(dataloader):
    metric = metrics.MetricCompose({
        'Acc': metrics.TopkAccuracy(),
        'Loss': metrics.RunningLoss(torch.nn.CrossEntropyLoss(reduction='sum'))
    })
    return Evaluator( metric, dataloader=dataloader )


def ddp_classification_evaluator(dataloader, task, num_classes, device):
    metric = torchmetrics.Accuracy(task=task, num_classes=num_classes).to(device)
    return DDP_Evaluator(metric, dataloader)


def advarsarial_classification_evaluator(dataloader, adversary):
    metric = metrics.MetricCompose({
        'Acc': metrics.TopkAccuracy(),
        'Loss': metrics.RunningLoss(torch.nn.CrossEntropyLoss(reduction='sum'))
    })
    return AdvEvaluator( metric, dataloader=dataloader, adversary=adversary)

def class_data_counts(dataloader):
    # targets = dataloader.dataset.imgs
    # labels = []
    # for _, label in dataloader:
        # labels.append(label.detach().cpu().numpy())
    # labels = np.concatenate(labels).astype(int)
    # for _, label in targets:
        # labels.append(label)
    labels = dataloader.dataset.labels
    labels = np.array(labels).astype(int)
    class_counts = []
    for l in np.unique(labels):
        class_counts.append(len(labels[labels == l]))
    total = np.sum(class_counts)
    print(f'class counts total:{total}')
    return class_counts

def evaluate_class_acc(model, valset, class_counts, device, name):
    preds, labels = [], []
    for i, (image, label) in enumerate(valset):
        # print(f'evaluating {i}th batch of data...')
        image, label = image.to(device), label.to(device)
        output = model(image)
        preds.append(output.detach().cpu())
        labels.append(label.detach().cpu())
    preds = torch.argmax(torch.cat(preds), dim=1)
    labels = torch.cat(labels)
    shot_accs, class_accs = shot_acc(preds, labels, class_counts)
    shot_nums = get_shot_num(class_counts)
    print(f'[Eval] {name} Class Accs: \n{class_accs}')
    print(f'[Eval] {name} Multi-shot Accs:' + \
          f'Many({shot_nums["many"]}): {shot_accs[0]:.4f}, ' + \
          f'Medium({shot_nums["medium"]}): {shot_accs[1]:.4f}, ' + \
          f'Few({shot_nums["few"]}): {shot_accs[2]:.4f}')
    top1 = torch.tensor(class_accs).mean()*100
    return shot_accs, torch.tensor(class_accs), top1

def shot_acc(preds, labels, train_class_counts):
    """ eval model performance on Many / Median / Low shot classes
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    many_shot_thr, low_shot_thr = get_shot_thld(train_class_counts)
    test_class_count, class_correct = [], []
    for l in np.unique(labels):
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())
    many_shot, median_shot, low_shot = [], [], []
    for i in range(len(train_class_counts)):
        if train_class_counts[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_counts[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))

    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)]
    return (np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)), class_accs