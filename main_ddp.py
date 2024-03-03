import os
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dataloader.dataset import MedicalDataSets
import albumentations as albu
from albumentations.core.composition import Compose

from utils.util import AverageMeter
import utils.losses as losses
from utils.metrics import iou_score

from network.CMUNeXt import cmunext, cmunext_s, cmunext_l

os.environ['LOCAL_RANK'] = "0"

def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_torch(41)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="CMUNeXt",
                    choices=["CMUNeXt", "CMUNeXt-S", "CMUNeXt-L"], help='model')
parser.add_argument('--base_dir', type=str, default="./data/busi", help='dir')
parser.add_argument('--train_file_dir', type=str, default="busi_train.txt", help='dir')
parser.add_argument('--val_file_dir', type=str, default="busi_val.txt", help='dir')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--img_ext', type=str, default=".jpg", help='dir')
parser.add_argument('--num_classes', type=int, default=1,
                    help='number of classes')

args = parser.parse_args()


def getDataloader():
    img_size = 256
    train_transform = Compose([
        albu.RandomRotate90(),
        albu.Flip(),
        albu.Resize(img_size, img_size),
        albu.Normalize(),
    ])

    val_transform = Compose([
        albu.Resize(img_size, img_size),
        albu.Normalize(),
    ])
    print(f'classes: {args.num_classes}')
    db_train = MedicalDataSets(base_dir=args.base_dir, split="train", transform=train_transform,
                               train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir, 
                               img_ext=args.img_ext, num_classes=args.num_classes)
    db_val = MedicalDataSets(base_dir=args.base_dir, split="val", transform=val_transform,
                             train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir, 
                             img_ext=args.img_ext, num_classes=args.num_classes)
    print("train num:{}, val num:{}".format(len(db_train), len(db_val)))
    
    # DDP: 使用 DistributedSampler, DDP model 会自动处理数据的分发
    train_sampler = torch.utils.data.distributed.DistributedSampler(db_train)
    val_sampler = torch.utils.data.distributed.DistributedSampler(db_val)
    # DDP: 需要注意的是，这里的 batch_size 指的是每个进程的 batch_size
    # 也就是说，总 batch_size 是 batch_size * world_size
    trainloader = DataLoader(db_train, batch_size=args.batch_size, num_workers=32, pin_memory=False, sampler=train_sampler)
    valloader = DataLoader(db_val, batch_size=args.batch_size, num_workers=32, sampler=val_sampler)
    
    return trainloader, valloader


def get_model(args):
    if args.model == "CMUNeXt":
        model = cmunext(num_classes=args.num_classes)
    elif args.model == "CMUNeXt-S":
        model = cmunext_s(num_classes=args.num_classes)
    elif args.model == "CMUNeXt-L":
        model = cmunext_l(num_classes=args.num_classes)
    else:
        model = None
        print("model err")
        exit(0)
    return model


def train(args):
    base_lr = args.base_lr
    
    # DDP: DDP backend 初始化
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    # 准备数据，要在DDP初始化之后进行 
    trainloader, valloader = getDataloader()
    
    # 构造模型
    model = get_model(args)
    model = model.to(local_rank)
    # DDP: 构造DDP模型
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    print("train file dir:{} val file dir:{}".format(args.train_file_dir, args.val_file_dir))
    
    # DDP：要在构造DDP model之后，才能用model初始化optimizer
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    criterion = losses.__dict__['BCEDiceLoss']().to(local_rank)
    
    print("{} iterations per epoch".format(len(trainloader)))
    best_iou = 0
    iter_num = 0
    max_epoch = 400
    max_iterations = len(trainloader) * max_epoch
    for epoch_num in range(max_epoch):
        # DDP: 设置sampler的epoch
        # DistributedSampler需要这个来指定shuffle方式
        # 通过维持各个进程之间的相同随机种子使不同进程能获得同样的shuffle效果
        trainloader.sampler.set_epoch(epoch_num)
        model.train()
        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter(),
                      'val_loss': AverageMeter(),
                      'val_iou': AverageMeter(),
                      'SE': AverageMeter(),
                      'PC': AverageMeter(),
                      'F1': AverageMeter(),
                      'ACC': AverageMeter()
                      }
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(local_rank), label_batch.to(local_rank)

            outputs = model(volume_batch)
            # print(f'outputs: {outputs.shape}')
            # print(f'label_batch: {label_batch.shape}')
            
            loss = criterion(outputs, label_batch)
            iou, dice, _, _, _, _, _ = iou_score(outputs, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            avg_meters['loss'].update(loss.item(), volume_batch.size(0))
            avg_meters['iou'].update(iou, volume_batch.size(0))

        model.eval()
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(valloader):
                input, target = sampled_batch['image'], sampled_batch['label']
                input = input.to(local_rank)
                target = target.to(local_rank)
                output = model(input)
                loss = criterion(output, target)
                
                iou, _, SE, PC, F1, _, ACC = iou_score(output, target)
                avg_meters['val_loss'].update(loss.item(), input.size(0))
                avg_meters['val_iou'].update(iou, input.size(0))
                avg_meters['SE'].update(SE, input.size(0))
                avg_meters['PC'].update(PC, input.size(0))
                avg_meters['F1'].update(F1, input.size(0))
                avg_meters['ACC'].update(ACC, input.size(0))

        print(
            'epoch [%d/%d]  train_loss : %.4f, train_iou: %.4f '
            '- val_loss %.4f - val_iou %.4f - val_SE %.4f - val_PC %.4f - val_F1 %.4f - val_ACC %.4f'
            % (epoch_num, max_epoch, avg_meters['loss'].avg, avg_meters['iou'].avg,
               avg_meters['val_loss'].avg, avg_meters['val_iou'].avg, avg_meters['SE'].avg,
               avg_meters['PC'].avg, avg_meters['F1'].avg, avg_meters['ACC'].avg))

        if dist.get_rank()==0 and avg_meters['val_iou'].avg > best_iou:
            if not os.path.exists('./checkpoint'):
                os.mkdir('checkpoint')
            torch.save(model.module.state_dict(), 'checkpoint/{}_model_{}.pth'
                       .format(args.model, args.train_file_dir.split(".")[0]))
            best_iou = avg_meters['val_iou'].avg
            print("=> saved best model")
    return "Training Finished!"


if __name__ == "__main__":
    train(args)
