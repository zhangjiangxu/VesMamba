import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
from dataset import VesselDataSet
import argparse
import logging
import time
import random
import numpy as np
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LinearLR
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from val import val_all_case
from utils import losses
import setproctitle
from networks.Edge_Scan_Segmamba_v3_GSLSConv_version2 import EdgeAwareScanSegMamba_v3_GSLSC_version2


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/segmamba', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='Edge_Scan_SegMamba_v3_mean_edge_GSLSC_version2', help='model_name')
parser.add_argument('--max_epoch', type=int,  default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.0001, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')


args = parser.parse_args()

proc_title = args.exp
setproctitle.setproctitle(proc_title)
train_data_path = args.root_path
snapshot_path = "/home/segmamba/model/" + args.exp + "/"
print("=================================")
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
print("GPU: ", args.gpu)
batch_size = args.batch_size
print("batch_size: ", batch_size)
base_lr = args.base_lr
print("base_lr: ", base_lr)
max_epoch = args.max_epoch
print("max_epoch: ", max_epoch)
print("=================================")

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (128, 128, 128)


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path, exist_ok=True)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    #model
    model = EdgeAwareScanSegMamba_v3_GSLSC_version2(in_chans=1,
                         out_chans=2,
                         depths=[1, 1, 1, 1],
                         feat_size=[48, 96, 192, 384]).cuda()

    x_transforms = transforms.ToTensor()
    y_transforms = transforms.ToTensor()

    db_train = VesselDataSet(mra_dir="/home/segmamba/MRA_data_preprocess/train_128_2000/data/",seg_dir="/home/segmamba/MRA_data_preprocess/train_128_2000/seg/", transform= x_transforms, target_transform= y_transforms)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    trainloader = DataLoader(db_train, batch_size=batch_size, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    print("Train dataset number of batch:", len(trainloader))
    
    #optimizer = optim.SGD(model.parameters(), lr=base_lr , momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=1e-5)
    
    
    total_epochs = max_epoch
    warmup_epochs = 10
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01, 
        end_factor=1.0, 
        total_iters=warmup_epochs
    )
    
    """
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs 
        
    )"""
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=0.000001)
   
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]  


    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    best_performance = 0
    print("max_epoch: {}".format(max_epoch))
    print("train start!")
    print("-----------------------------")
    model.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        iter_num = 0
        epoch_id = epoch_num + 1
        time1 = time.time()
        for batch_idx, (img_x, img_y, x_path) in enumerate(trainloader):
            optimizer.zero_grad()
            image = img_x.cuda()
            label = img_y.long().cuda()
            outputs = model(image)
            outputs_soft = F.softmax(outputs, dim=1)
            ## calculate the loss
            ce_loss = F.cross_entropy(outputs, label.squeeze(1))
            dice_loss = losses.dice_loss1(outputs_soft[:, 1, :, :, :], label)
            loss = 0.5 * ce_loss + 0.5 * dice_loss


            loss.backward()
            optimizer.step()
            iter_num = iter_num + 1

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('lr', current_lr, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/ce_loss', ce_loss, iter_num)
            writer.add_scalar('loss/dice_loss', dice_loss, iter_num)
            logging.info('iteration %d/%d : loss : %f, ce_loss: %f, dice_loss: %f, lr: %.6f' %
                         (iter_num, len(trainloader), loss.item(), ce_loss.item(), dice_loss.item(), current_lr))
        time2 = time.time()  
        train_time = time2 - time1

        print(f'training time: ', train_time)
        break

        print("val start!")
        model.eval()
        with torch.no_grad():
            avg_metric = val_all_case(net=model, val_path='/home/segmamba/MRA_data_preprocess/val_128_2000/',
                                      num_classes=2,
                                      patch_size=patch_size, stride_xy=64, stride_z=64)

            save_mode_path = os.path.join(snapshot_path, 'epoch_{}_dice_{}.pth'.format(epoch_id, avg_metric))
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            logging.info('epoch %d : dice_score : %f ' % (epoch_id, avg_metric))
            if avg_metric > best_performance:
                best_performance = avg_metric
                save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.exp))
                torch.save(model.state_dict(), save_best)
            print('best_performance of dice is {}'.format(best_performance))
        model.train()
        scheduler.step()
    writer.close()
