import os
import copy
import time
import random
import registry
import datafree
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import configs
from registry import DATASET_INFO


best_acc1 = 0
def main():
    begin = time.time()
    args = configs.args
    port =  random.randint(20000, 29999)
    args.dist_url =f'tcp://localhost:{port}'
    print(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    args.ngpus_per_node = ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # multi-node
        args.world_size = ngpus_per_node * args.world_size
        args.batch_size //= args.world_size
        args.workers //= args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # single-gpu
        main_worker(args.gpu, ngpus_per_node, args)
    end = time.time()
    print(f'Finished! Time elspased:{int(end-begin)}s')



def main_worker(gpu, ngpus_per_node, args):

    main_start = time.time()
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
    args.is_rank0 = (args.rank <= 0)
    ############################################
    # Logger
    ############################################
    exp_name = f'{args.dataset}-{args.teacher}-{args.student}'
    log_name = f'R{args.rank}-{exp_name}-{args.log_tag}' \
                    if args.multiprocessing_distributed else exp_name
    ckpt_root = f'checkpoints/datafree-{args.method}'
    log_txt = f'log-{exp_name}-{args.log_tag}.txt'
    output =  os.path.join(ckpt_root, log_txt)
    # screen logger
    args.logger = datafree.utils.logger.get_logger(log_name, output)
    if args.is_rank0:
        for k, v in datafree.utils.flatten_dict( vars(args) ).items():
            args.logger.info( "%s: %s"%(k,v) )
    # tensorboard logger
    tb = datafree.utils.TBSummary(os.path.join(ckpt_root, exp_name))

    ############################################
    # Setup models
    ############################################
    def prepare_model(model):
        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
            return model
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
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
                return model
            else:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)
                return model
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
            return model
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            model = torch.nn.DataParallel(model).cuda()
            return model

    num_classes = DATASET_INFO[args.dataset]['num_classes']
    student = registry.get_model(args.dataset, args.student, num_classes=num_classes)
    teacher = registry.get_model(args.dataset, args.teacher, num_classes=num_classes, pretrained=True).eval()
    args.normalizer = datafree.utils.Normalizer(**registry.NORMALIZE_DICT[args.dataset])
    args.teacher_path = os.path.expanduser(args.teacher_path)
    teacher_path = args.teacher_path or f'checkpoints/pretrained/{args.dataset}_{args.teacher}.pth'
    args.logger.info(f'Loading pretrained teacher model from:{teacher_path}')
    try:
        teacher.load_state_dict(torch.load(teacher_path ,map_location='cpu')['state_dict'])
    except:
        teacher.load_state_dict(torch.load(teacher_path ,map_location='cpu'))
    student = prepare_model(student)
    teacher = prepare_model(teacher)

    ############################################
    # Setup dataset
    ############################################
    # train dataset was only used to calculate the data distribution
    _, train_dataset, val_dataset = registry.get_dataset(args.dataset,
                    args.data_root, imbalance_ratio=args.imbalance_ratio)

    sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if args.distributed else None

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=sampler)

    if args.distributed:
        evaluator = datafree.evaluators.ddp_classification_evaluator(val_loader,
                                task="multiclass", num_classes=num_classes, device=args.gpu)
    else:
        evaluator = datafree.evaluators.classification_evaluator(val_loader)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    train_class_counts = datafree.evaluators.class_data_counts(train_loader)
    # train_class_counts = datafree.evaluators.class_data_counts(val_loader)

    # eval pretrained teacher
    if args.eval_teacher:
        print('[Eval] Start evaluating teacher model...')
        start = time.time()
        teacher.eval()
        if 'LT' in args.dataset:
            shot_accs, class_accs, top1 = datafree.evaluators.evaluate_class_acc(teacher,
            val_loader, train_class_counts, args.gpu, 'Teacher')
            print(f'[Eval] Teacher Top1 Acc={top1:.4f}')
            print(f'[Eval] Shot Accs={shot_accs}')
        else:
            eval_results = evaluator(teacher, args.gpu)
            top1, _ = eval_results['Acc']
            print(f'[Eval] Teacher Top1 Acc={top1:.4f}')
        end = time.time()
        print(f'[Eval] Time elapsed:{int(end-start)}s')


    criterion = datafree.criterions.KLDiv(T=args.T)


    ############################################
    # Setup data-free synthesizers
    ############################################
    if args.synthesis_batch_size is None:
        args.synthesis_batch_size = args.batch_size

    if args.method=='deepinv':
        synthesizer = datafree.synthesis.DeepInvSyntheiszer(
                 teacher=teacher, student=student,
                 img_size=DATASET_INFO[args.dataset]['image_size'],
                 train_class_counts=train_class_counts,
                 num_classes=num_classes, iterations=args.g_steps, lr_g=args.lr_g,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size,
                 adv=args.adv, bn=args.bn, oh=args.oh, tv=0.001, l2=0.0,
                 save_dir=args.save_dir, transform=train_dataset.transform,
                 normalizer=args.normalizer, device=args.gpu)

    elif args.method in ['zskt', 'dfad', 'dfq', 'dafl', 'mad', 'ipad']:
        # nz = 512 if args.method=='dafl' else 256
        nz = DATASET_INFO[args.dataset]['noise_dim']
        nl = nz if args.generator == 'condgan' else 0 # noise dim of conditional generator
        if args.dataset in ['imagenet-LT', 'places-LT']:
            generator = datafree.models.generator.HugeGenerator(nz=nz, ngf=64,
                        img_size=DATASET_INFO[args.dataset]['image_size'][1], nc=3, nl=nl)
        else:
            generator = datafree.models.generator.LargeGenerator(nz=nz, ngf=64,
                        img_size=DATASET_INFO[args.dataset]['image_size'][1], nc=3, nl=nl)
        generator = prepare_model(generator)
        criterion = torch.nn.L1Loss() if args.method=='dfad' else datafree.criterions.KLDiv(reduction='none')
        synthesizer = datafree.synthesis.GenerativeSynthesizer(
                teacher=teacher, student=student, generator=generator, nz=nz,
                img_size=DATASET_INFO[args.dataset]['image_size'],
                train_class_counts=train_class_counts,
                num_classes=num_classes, iterations=args.g_steps, lr_g=args.lr_g,
                synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size,
                adv=args.adv, bn=args.bn, oh=args.oh, rw=args.rw, act=args.act,
                balance=args.balance, decorr=args.decorr, criterion=criterion,
                use_fp16=args.fp16, autocast=args.autocast, scaler=args.scaler,
                normalizer=args.normalizer, device=args.gpu)
        if args.method == 'mad':
            mmt_synthesizer = copy.deepcopy(synthesizer)
            mmt_synthesizer.sample_batch_size //= 2
            synthesizer.sample_batch_size //= 2
    elif args.method=='cmi':
        nz = DATASET_INFO[args.dataset]['noise_dim']
        nl = nz if args.generator == 'condgan' else 0 # conditional generator
        if args.dataset in ['imagenet-LT', 'places-LT']:
            generator = datafree.models.generator.HugeGenerator(nz=nz, ngf=512,
                        img_size=DATASET_INFO[args.dataset]['image_size'][1], nc=3, nl=nl)
        else:
            generator = datafree.models.generator.LargeGenerator(nz=nz, ngf=64,
                        img_size=DATASET_INFO[args.dataset]['image_size'][1], nc=3, nl=nl)
        generator = prepare_model(generator)
        feature_layers = None # use all conv layers
        if args.teacher=='resnet34': # only use blocks
            feature_layers = [teacher.layer1, teacher.layer2, teacher.layer3, teacher.layer4]
        synthesizer = datafree.synthesis.CMISynthesizer(teacher, student, generator,
                 nz=nz, img_size=DATASET_INFO[args.dataset]['image_size'],
                 train_class_counts=train_class_counts,
                 num_classes=num_classes,
                 # if feature layers==None, all convolutional layers will be used by CMI.
                 feature_layers=feature_layers, bank_size=10240, n_neg=1024,
                 head_dim=256, init_dataset=args.cmi_init,
                 iterations=args.g_steps, lr_g=args.lr_g, progressive_scale=False,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size,
                 adv=args.adv, bn=args.bn, oh=args.oh, cr=args.cr, cr_T=args.cr_T,
                 save_dir=args.save_dir, transform=train_dataset.transform,
                 normalizer=args.normalizer, device=args.gpu)
    else:
        raise NotImplementedError

    ############################################
    # Setup optimizer and lr schduler for student
    ############################################
    args.logger.info('Preparing Scheduler...')
    optimizer = torch.optim.SGD(student.parameters(), args.lr, weight_decay=args.weight_decay, momentum=0.9)
    if args.scheduler == 'step':
        milestones = [ int(ms) for ms in args.lr_decay_milestones.split(',') ]
        scheduler = torch.optim.lr_scheduler.MultiStepLR( optimizer, milestones=milestones, gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=args.epochs)

    ############################################
    # Resume
    ############################################
    args.current_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume, map_location='cpu')
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)

            if isinstance(student, nn.Module):
                student.load_state_dict(checkpoint['state_dict'])
            else:
                student.module.load_state_dict(checkpoint['state_dict'])
            best_acc1 = checkpoint['best_acc1']
            try:
                args.start_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
            except: print("Fails to load additional model information")
            print("[!] loaded checkpoint '{}' (epoch {} acc {})"
                  .format(args.resume, checkpoint['epoch'], best_acc1))
        else:
            print("[!] no checkpoint found at '{}'".format(args.resume))

    ############################################
    # Evaluate
    ############################################
    if args.evaluate_only:
        print(f'start evaluating student model: {args.student}...')
        student.eval()
        eval_results = evaluator(student, device=args.gpu)
        print('[Eval] Student Acc={acc:.4f}'.format(acc=eval_results['Acc'][0]))
        '''
        shot_accs, class_accs, top1 = datafree.evaluators.evaluate_class_acc(student,
                        val_loader, train_class_counts, args.gpu, 'Student')
        '''
        return

    ############################################
    # Train Loop
    ############################################
    shot_accs_his, s_shot_kl_his, gen_shot_kl_his = [], [], []
    t_argmax_his, s_argmax_his = [], []
    args.logger.info('Start Training Loop')
    for epoch in range(args.start_epoch, args.epochs):
        t_start = time.time()
        adv_start = time.time()
        args.current_epoch=epoch
        for step in range( args.ep_steps//args.kd_steps ): # total kd_steps < ep_steps
            ### 1. Data synthesis
            syn_results = synthesizer.synthesize(epoch, step) # g_steps
            if args.method == 'mad':
                _ = mmt_synthesizer.momentum_update(synthesizer.generator.state_dict()) # g_steps

            ### 2. Knowledge distillation
            if args.method == 'mad':
                kd_results = kd_train(mmt_synthesizer, [student, teacher], criterion,
                            optimizer, args, mmt_synthesizer) # kd_steps
            else:
                kd_results = kd_train(synthesizer, [student, teacher], criterion,
                            optimizer, args,) # kd_steps

        # evaluation
        adv_end = time.time()
        e_start = time.time()
        student.eval()
        shot_accs, class_accs, top1 = datafree.evaluators.evaluate_class_acc(student,
                            val_loader, train_class_counts, args.gpu, 'Student')
        eval_results = evaluator(student, args.gpu)
        top1, _ = eval_results['Acc']
        e_end = time.time()
        s_start = time.time()
        # if args.plot and args.is_rank0:
        if args.plot:
            if epoch % 1 == 0 or epoch == args.epochs-1:
                for vis_name, vis_image in syn_results.items():
                    if vis_name == 'synthetic':
                        with args.autocast(enabled=args.fp16):
                            output = teacher(vis_image)
                        pred = torch.argmax(output, dim=1).detach().cpu()
                        save_path = f'{ckpt_root}/synthetic_images/rank{args.rank}/{vis_name}_{epoch}_{args.log_tag}.png'
                        datafree.utils.save_image_batch(vis_image, save_path, pred=pred, pack=True)
            args.logger.info(f'synthetic images saved!')

        # plots and logs
        if args.tensorboard:
            # log data
            tb.scalar_summary('eval/student_acc', top1, epoch)
            tb.scalar_summary('synthetic/adv_loss', syn_results['loss_adv_total'], epoch)
            tb.scalar_summary('synthetic/bn_loss', syn_results['loss_bn_total'], epoch)
            tb.scalar_summary('synthetic/oh_loss', syn_results['loss_oh_total'], epoch)
            tb.scalar_summary('synthetic/act_loss', syn_results['loss_act_total'], epoch)
            tb.scalar_summary('synthetic/balance_loss', syn_results['loss_balance_total'], epoch)
            tb.scalar_summary('synthetic/decorr_loss', syn_results['loss_decorr_total'], epoch)
            tb.scalar_summary('distillation/student_loss', kd_results['s_loss_total'], epoch)
            tb.histo_summary('distillation/teacher_argmax_dist', kd_results['t_argmax'], epoch, bins=num_classes)
            tb.histo_summary('distillation/student_argmax_dist', kd_results['s_argmax'], epoch, bins=num_classes)
            if args.method == 'cmi':
                tb.scalar_summary('synthetic/contrastive_loss', syn_results['loss_act_total'], epoch)

            t_argmax_his.append(kd_results['t_argmax'])
            s_argmax_his.append(kd_results['s_argmax'])

            # save all logs
            if epoch == args.epochs-1:
                log_data = {'t_shot_argmax_his':t_argmax_his,
                            's_shot_argmax_his':s_argmax_his,
                            's_shot_kl_his':s_shot_kl_his,
                            'shot_accs_his':shot_accs_his,
                            'gen_shot_kl_his':gen_shot_kl_his,
                            }
                torch.save(log_data, os.path.join(ckpt_root, f'{args.method}-{exp_name}-{args.log_tag}-log_data.pth'))
            args.logger.info(f'tensorboard logs saved!')

        s_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        is_best = bool(top1 > best_acc1)
        best_acc1 = max(top1, best_acc1)
        _best_ckpt = f'{ckpt_root}/{exp_name}.pth'
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.is_rank0):
            datafree.utils.save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.student,
            'state_dict': student.state_dict(),
            'best_acc1': float(best_acc1),
            'optimizer' : optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            }, is_best, _best_ckpt)
        t_end = time.time()
        s_end = time.time()
        args.logger.info(f'[Eval] Epoch={args.current_epoch} ' + \
                     f'Acc@1={top1:.4f} S_Lr={s_lr:.4f} ' + \
                     f'Time={int(t_end-t_start)}s ' + \
                     f'ADV:{int(adv_end-adv_start)}s Eval:{int(e_end-e_start)}s ' +\
                     f'Many:{(100*shot_accs[0]):.2f} Medium:{(100*shot_accs[1]):.2f} Few:{(100*shot_accs[2]):.2f} ' +\
                     f'Save:{int(s_end-s_start)}s ' +\
                     f'Best_model={str(is_best)}')
    main_end = time.time()
    args.logger.info( f'Best: {best_acc1:.4f} ' +\
                      f'Total Time:{int(main_end - main_start)}s'
                        )

def kd_train(synthesizer, models, criterion, optimizer, args, ext_synthesizer=None):
    k_start = time.time()
    loss_metric = datafree.metrics.RunningLoss(datafree.criterions.KLDiv(reduction='sum'))
    acc_metric = datafree.metrics.TopkAccuracy(topk=(1,5))
    student, teacher = models
    optimizer = optimizer
    student.train()
    teacher.eval()
    s_loss_total = 0.
    t_argmax, s_argmax = [], []
    t_preds, s_preds = [], []
    for i in range(args.kd_steps):
        images = synthesizer.sample()
        # print(f'debug images shape:{images.shape}')
        if ext_synthesizer is not None:
            ext_images = ext_synthesizer.sample()
            images = torch.cat([images, ext_images], dim=0)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        with args.autocast(enabled=args.fp16):
            with torch.no_grad():
                t_out, t_feat = teacher(images, return_features=True)
            s_out = student(images.detach())
            loss_s = criterion(s_out, t_out.detach()).sum() / s_out.size(0)
        optimizer.zero_grad()
        if args.fp16:
            scaler = args.scaler
            scaler.scale(loss_s).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_s.backward()
            optimizer.step()
        acc_metric.update(s_out, t_out.max(1)[1])
        loss_metric.update(s_out, t_out)
        t_argmax.append(torch.argmax(t_out, 1).detach().cpu())
        s_argmax.append(torch.argmax(s_out, 1).detach().cpu())
        t_preds.append(t_out.detach().cpu())
        s_preds.append(s_out.detach().cpu())
        if args.print_freq>0 and i % args.print_freq == 0:
            (train_acc1, train_acc5), train_loss = \
                acc_metric.get_results(), loss_metric.get_results()
            lr=optimizer.param_groups[0]['lr']
            args.logger.info(f'[Train] Epoch={args.current_epoch}, ' + \
                                f'Iter={i}/{len(args.kd_steps)}, ' + \
                                f'train_acc@1={train_acc1:.4f}, ' + \
                                f'train_acc@5={train_acc5:.4f}, ' + \
                                f'train_Loss={train_loss:.4f}, ' + \
                                f'Lr={lr:.4f}')
            loss_metric.reset(), acc_metric.reset()
    s_loss_total += loss_metric.get_results()
    k_end = time.time()
    return {
            'time': int(k_end - k_start),
            's_loss_total': s_loss_total,
            't_argmax': torch.cat(t_argmax),
            's_argmax': torch.cat(s_argmax),
    }


if __name__ == '__main__':
    main()