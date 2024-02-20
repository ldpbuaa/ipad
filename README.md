# Impartial Adversarial Distillation
This file provides the commands for reproducing results in this work.
## Pretrain the Teacher Model
### Pretrain on Imbalanced Dataset
```
python pretrain.py \
--data_root=<data_path> \
--model=<model> \
--dataset=<dataset_name> \
--sampler=None \
--lr=0.1 \
--scheduler='step' \
--epochs=200 \
--batch-size=256 \
--lr-decay-milestones=80,120,160 \
--imbalance_ratio=<imbalance_ratio> \
--multiprocessing-distributed=False \
--rank=0 \
--world-size=0
```
### Rebalanced Finetune
```
python pretrain.py \
--data_root=<data_path> \
--dataset=<dataset_name> \
--sampler='class-aware' \
--resume=<check_point_path> \
--epochs=20 \
--batch-size=256 \
--reset-optim=True \
--freeze-backbone=True \
--learning-rate=0.01 \
--weight-decay=4e-5 \
--lr-decay-milestones=10 \
--imbalance_ratio=<imbalance_ratio> \
--multiprocessing-distributed=False \
--rank=0 \
--world-size=0
```

## Data-free Knowledge Distillation
```
python main.py \
--method <method> \
--dataset <dataset_name> \
--batch_size 256 \
--scheduler cos \
--teacher <teacher_model> \
--teacher_path <teacher_checkpoint_path> \
--student <student_model> \
--imbalance_ratio <imbalance_ratio> \
--lr <student_learning_rate> \
--epochs 150 \
--ep_steps 1000 \
--kd_steps 10 \
--g_steps 1 \
--lr_g 1e-3 \
--adv <adv_loss_coefficient> \
--T 1 \
--bn <bn_loss_coefficient> \
--oh <one_hot_loss_coefficient> \
--act <activation_loss_coefficient> \
--balance <balance_loss_coefficient> \
--rw <reweight_loss_coefficient> \
--gpu 0
```


# Acknowledgement
This code implementation is based on the code from the following papers:
- Contrastive Model Inversion for Data-Free Knowledge Distillation
- Decoupling Representation and Classifier for Long-Tailed Recognition

We thank the authors for making their code publicly avaliable to research community.
