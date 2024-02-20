import argparse
import os
import random
import time
import warnings
import ast

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data.distributed import DistributedSampler
from datafree.utils.misc import getStat

import registry
import datafree
from datafree.utils.misc import freeze_backbone
from registry import DATASET_INFO

parser = argparse.ArgumentParser(description='Pretraining Models')
# Basic Settings
parser.add_argument('--data_root', default='data')
parser.add_argument('--model', default='wrn40_2',)
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--save_name', default='')
parser.add_argument('--imbalance_ratio', type=float, default=1.)
parser.add_argument('--sampler', default='', choices=['class-aware', 'uniform'])
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model from public repo')
# Optimization
parser.add_argument('-e', '--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--optimizer', default='SGD', type=str, choices=['SGD', 'Adam'],
                    help='lr scheduer')
parser.add_argument('--scheduler', default='cos', type=str, choices=['cos', 'step'],
                    help='lr scheduer')
parser.add_argument('-ldm', '--lr_decay_milestones', default="120,150,180", type=str,
                    help='milestones for learning rate decay')
parser.add_argument('-m', '--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('-wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-g', '--gpu', default=0, type=int,
                    help='GPU id to use.')
# Evaluation
parser.add_argument('--evaluate_only', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--reset_optim', type=ast.literal_eval,
                    help='reset optimization settings: start-epoch, optimizer, scheduler')
parser.add_argument('--freeze_backbone', type=ast.literal_eval,
                    help='freeze model backbone when finetune')
# Device & FP16
parser.add_argument('--fp16', type=ast.literal_eval,
                    help='use fp16')
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('-r', '--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://localhost:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing_distributed', type=ast.literal_eval,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# Misc
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-p', '--print_freq', default=0, type=int,
                    metavar='N', help='print frequency (default: 0)')
parser.add_argument('-s', '--seed', default=None, type=int,
                    help='seed for initializing training.')

best_acc1 = 0

def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # multi-nodes
        args.world_size = ngpus_per_node * args.world_size
        args.batch_size //= args.world_size
        args.workers //= args.world_size
        # spawn main_worker
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # single GPU training
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    ############################################
    # GPU and FP16
    ############################################
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    if args.fp16:
        from torch.cuda.amp import autocast, GradScaler
        args.scaler = GradScaler()
        args.autocast = autocast
    else:
        args.autocast = datafree.utils.dummy_ctx
        args.scaler = None

    ############################################
    # Logger
    ############################################
    log_name = 'R%d-%s-%s'%(args.rank, args.dataset, args.model) if args.multiprocessing_distributed else f'{args.dataset}-{args.model}'
    args.logger = datafree.utils.logger.get_logger(log_name, output=f'checkpoints/scratch/log-{args.dataset}-{args.model}.txt')
    if args.rank<=0:
        for k, v in datafree.utils.flatten_dict( vars(args) ).items(): # print args
            args.logger.info( "%s: %s"%(k,v) )

    ############################################
    # Setup models
    ############################################
    # models
    num_classes = DATASET_INFO[args.dataset]['num_classes']
    model = registry.get_model(args.dataset, args.model, num_classes=num_classes, pretrained=args.pretrained)
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
    ############################################
    # Setup datasets
    ############################################
    # datasets
    _, train_dataset, val_dataset = registry.get_dataset(name=args.dataset,
                data_root=args.data_root, imbalance_ratio=args.imbalance_ratio)
    cudnn.benchmark = True
    # getStat(train_dataset)
    # getStat(val_dataset)
    # dataset sampler
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    elif args.sampler == 'class-aware' and not args.distributed:
        print('Using class-aware sampler')
        train_sampler = datafree.datasets.get_sampler()(train_dataset, num_samples_cls=4)
        val_sampler = None
    else:
        train_sampler = None
        val_sampler = None
    # dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    if args.distributed:
    # if False:
        evaluator = datafree.evaluators.ddp_classification_evaluator(val_loader,
                                    task="multiclass", num_classes=num_classes, device=args.gpu)
    else:
        evaluator = datafree.evaluators.classification_evaluator(val_loader)
    args.current_epoch = 0

    ############################################
    # Setup optimizer
    ############################################
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                        momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), args.lr)

    if args.scheduler == 'step':
        milestones = [ int(ms) for ms in args.lr_decay_milestones.split(',') ]
        scheduler = torch.optim.lr_scheduler.MultiStepLR( optimizer, milestones=milestones, gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=args.epochs)

    ############################################
    # Resume
    ############################################
    if args.resume:
        args.resume = os.path.expanduser(args.resume)
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)

            # if isinstance(model, nn.Module):
            #     model.load_state_dict(checkpoint['state_dict'])
            # else:
            #     model.module.load_state_dict(checkpoint['state_dict'])
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except:
                model.module.load_state_dict(checkpoint['state_dict'])

            try:
                if not args.reset_optim:
                    args.start_epoch = checkpoint['epoch']
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    scheduler.load_state_dict(checkpoint['scheduler'])
                    best_acc1 = checkpoint['best_acc1']
            except:
                print("Fails to load additional information")
            resume_acc1 = checkpoint['best_acc1']
            print("[!] loaded checkpoint '{}' (epoch {} acc {})"
                  .format(args.resume, checkpoint['epoch'], resume_acc1))
        else:
            print("[!] no checkpoint found at '{}'".format(args.resume))

        if args.freeze_backbone:
            print('[!] Freezing model backbone...')
            model = freeze_backbone(model)

    ############################################
    # Evaluate
    ############################################
    if args.evaluate_only:
        start = time.time()
        print(f'start evaluating model: {args.model}...')
        model.eval()
        eval_results = evaluator(model, device=args.gpu)
        eval_time = time.time() - start
        (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
        print('[Eval] Acc@1={acc1:.4f} Acc@5={acc5:.4f} Loss={loss:.4f} Eval Time:{eval_time:d}s'.format(
                                        acc1=acc1, acc5=acc5, loss=val_loss, eval_time=eval_time))
        return

    ############################################
    # Train Loop
    ############################################
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        args.current_epoch=epoch
        start = time.time()
        train(train_loader, model, criterion, optimizer, args)
        end = time.time()
        args.logger.info(f'Training Time Elapsed: {(end-start):.1f}s')
        start = time.time()
        model.eval()
        eval_results = evaluator(model, device=args.gpu)
        (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
        is_best = acc1 > best_acc1
        end = time.time()
        lr = optimizer.param_groups[0]['lr']
        args.logger.info(f'[Eval] Epoch={args.current_epoch} Acc@1={acc1:.4f} Acc@5={acc5:.4f} Loss={val_loss:.4f} Lr={lr:.4f} Eval Time:{(end-start):.1f}s Best_model={str(is_best)}')
                #.format(current_epoch=args.current_epoch, acc1=acc1, acc5=acc5, loss=val_loss, lr=optimizer.param_groups[0]['lr'], best_model=str(is_best) ))
        scheduler.step()
        best_acc1 = max(acc1, best_acc1)
        _best_ckpt = os.path.expanduser(args.save_name) or f'checkpoints/scratch/{args.dataset}_{args.model}.pth'
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            if isinstance(model, nn.parallel.DistributedDataParallel) or isinstance(model, nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model,
                'state_dict': state_dict,
                'best_acc1': float(best_acc1),
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best, _best_ckpt)
    if args.rank<=0:
        args.logger.info("Best: %.4f"%best_acc1)


def train(train_loader, model, criterion, optimizer, args):
    global best_acc1
    loss_metric = datafree.metrics.RunningLoss(nn.CrossEntropyLoss(reduction='sum'))
    acc_metric = datafree.metrics.TopkAccuracy(topk=(1,5))
    model.train()
    for i, (images, target) in enumerate(train_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)
        with args.autocast(enabled=args.fp16):
            output = model(images)
            loss = criterion(output, target)
        # measure accuracy and record loss
        acc_metric.update(output, target)
        loss_metric.update(output, target)
        optimizer.zero_grad()
        if args.fp16:
            scaler = args.scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.print_freq>0 and i % args.print_freq == 0:
            (train_acc1, train_acc5), train_loss = acc_metric.get_results(), loss_metric.get_results()
            args.logger.info('[Train] Epoch={current_epoch} Iter={i}/{total_iters}, train_acc@1={train_acc1:.4f}, train_acc@5={train_acc5:.4f}, train_Loss={train_loss:.4f}, Lr={lr:.4f}'
              .format(current_epoch=args.current_epoch, i=i, total_iters=len(train_loader), train_acc1=train_acc1, train_acc5=train_acc5, train_loss=train_loss, lr=optimizer.param_groups[0]['lr']))
            loss_metric.reset(), acc_metric.reset()


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)

if __name__ == '__main__':
    main()