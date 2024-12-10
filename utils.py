
import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from kmeans_pytorch import kmeans
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn import metrics
from torchvision.utils import make_grid
from einops import rearrange
from PIL import Image

Softmax_m = nn.Softmax(dim=1)
device = torch.device('cuda:0')

def device_ids_return(cuda_list):
    if  len(cuda_list) == 1:
        device_ids = [0]
    elif len(cuda_list) == 2:
        device_ids = [0,1]
    elif len(cuda_list) == 3:
        device_ids = [0,1,2]
    elif len(cuda_list) == 4:
        device_ids = [0,1,2,3]
    elif len(cuda_list) == 5:
        device_ids = [0,1,2,3,4]
    return device_ids

def findLastCheckpoint(save_dir):
    if os.path.exists(save_dir):
        file_list = os.listdir(save_dir)
        result = 0
        for file in file_list:
            try:
                num = int(file.split('.')[0].split('_')[-1])
                result = max(result, num)
            except:
                continue
        return result
    else:
        os.mkdir(save_dir)
        return 0

def tb_writer_display(writer, iter_num, lr_scheduler, epoch, seg_accu, binary_loss,
                      loss_1, loss_2, loss_3, loss_4):
    writer.add_scalar('Train/seg_accu', seg_accu, iter_num)
    writer.add_scalar('Train/binary_map_loss', binary_loss, iter_num)
    writer.add_scalar('Train/loss_1', loss_1, iter_num)
    writer.add_scalar('Train/loss_2', loss_2, iter_num)
    writer.add_scalar('Train/loss_3', loss_3, iter_num)
    writer.add_scalar('Train/loss_4', loss_4, iter_num)
    for count, gp in enumerate(lr_scheduler.optimizer.param_groups,1):
        writer.add_scalar('progress/lr_%d'%count, gp['lr'], iter_num)
    writer.add_scalar('progress/epoch', epoch, iter_num)
    writer.add_scalar('progress/curr_patience',lr_scheduler.num_bad_epochs,iter_num)
    writer.add_scalar('progress/patience',lr_scheduler.patience,iter_num)


def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)
        print(f"Making folder {folder_name}.")
    else:
        print(f"Folder {folder_name} exists.")

def class_weight(mask, mask_idx):
    mask_balance = torch.ones_like(mask).to(torch.float)
    if (mask == 1).sum():
        mask_balance[mask == 1] = 0.5 / ((mask == 1).sum().to(torch.float) / mask.numel())
        mask_balance[mask == 0] = 0.5 / ((mask == 0).sum().to(torch.float) / mask.numel())
    else:
        pass

    return mask.to(device), mask_balance.to(device)

def setup_optimizer(args, SegNet, FENet):
    params_dict_list = []
    params_dict_list.append({'params' : SegNet.module.parameters(), 'lr' : args.learning_rate})
    freq_list = []
    para_list = []
    for name, param in FENet.named_parameters():
        if 'fre' in name:
            freq_list += [param]
        else:
            para_list += [param]
    params_dict_list.append({'params' : freq_list, 'lr' : args.learning_rate*args.lr_backbone})
    params_dict_list.append({'params' : para_list, 'lr' : args.learning_rate})

    optimizer    = torch.optim.Adam(params_dict_list, lr=args.learning_rate*0.75, weight_decay=1e-06)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.step_factor, min_lr=1e-08,
                                     patience=args.patience, verbose=True)

    return optimizer, lr_scheduler

def restore_weight_helper(model, model_dir, initial_epoch):
    try:
        weight_path = '{}/{}.pth'.format(model_dir, initial_epoch)
        state_dict = torch.load(weight_path, map_location='cuda:0')['model']
        model.load_state_dict(state_dict)
        print('{} weight-loading succeeds: {}'.format(model_dir, weight_path))
    except:
        print('{} weight-loading fails'.format(model_dir))

    return model

def restore_optimizer(optimizer, model_dir, initial_epoch):
    try:
        weight_path = '{}/{}.pth'.format(model_dir, initial_epoch)
        state_dict = torch.load(weight_path, map_location='cuda:0')
        print('Optimizer weight-loading succeeds.')
        optimizer.load_state_dict(state_dict['optimizer'])
    except:
        pass
    return optimizer

def composite_obj(args, loss, loss_1, loss_2, loss_3, loss_4, loss_binary):
    if args.ablation == 'full':
        loss_total = 100*loss + loss_1 + loss_2 + loss_3 + 100*loss_4 + loss_binary
    elif args.ablation == 'base':
        loss_total = loss_4
    elif args.ablation == 'fg':
        loss_total = loss_1 + loss_2 + loss_3 + loss_4
    elif args.ablation == 'local':
        loss_total = loss + 10e-6*(loss_1 + loss_2 + loss_3 + loss_4)
    else:
        assert False
    return loss_total

def composite_obj_step(args, loss_4_sum, map_loss_sum):
    if args.ablation == 'full':
        schedule_step_loss = loss_4_sum + map_loss_sum
    elif args.ablation == 'base':
        schedule_step_loss = loss_4_sum
    elif args.ablation == 'fg':
        schedule_step_loss = loss_4_sum
    elif args.ablation == 'local':
        schedule_step_loss = map_loss_sum
    else:
        assert False
    return schedule_step_loss

def viz_log(args, mask, pred_mask, image, iter_num, step, mode='train'):
    mask = torch.unsqueeze(mask, dim=1)
    pred_mask = torch.unsqueeze(pred_mask, dim=1)
    mask_viz = torch.cat([mask]*3, axis=1)
    pred_mask = torch.cat([pred_mask]*3, axis=1)
    image = torch.nn.functional.interpolate(image,
                                          size=(256, 256), 
                                          mode='bilinear')
    fig_viz = torch.cat([mask_viz, image, pred_mask], axis=0)
    grid = make_grid(fig_viz, nrow=mask_viz.shape[0])
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    img_h = Image.fromarray(grid.astype(np.uint8))
    os.makedirs(f"./viz_{mode}/", exist_ok=True)
    if mode == 'train':
        img_h.save(f"./viz_{mode}/iter_{iter_num}.jpg")
    else:
        img_h.save(f"./viz_{mode}/iter_{iter_num}_step_{step}.jpg")

def train_log_dump(args, seg_correct, seg_total, binary_map_loss_sum, loss_1_sum, loss_2_sum, loss_3_sum,
                    loss_4_sum, epoch, step, writer, iter_num, lr_scheduler):

    seg_accu = seg_correct / seg_total * 100

    binary_loss  = binary_map_loss_sum / args.dis_step
    loss_1 = loss_1_sum / args.dis_step
    loss_2 = loss_2_sum / args.dis_step
    loss_3 = loss_3_sum / args.dis_step
    loss_4 = loss_4_sum / args.dis_step
    print(f'[Epoch: {epoch+1}, Step: {step + 1}] seg_acc: {seg_accu:.2f}')
    print(f'cls1_loss: {loss_1:.3f}, cls2_loss: {loss_2:.3f}, cls3_loss: {loss_3:.3f}, '+
          f'cls4_loss: {loss_4:.3f}, binary_map_loss: {binary_loss:.3f}')
    '''write the tensorboard.'''
    tb_writer_display(writer, iter_num, lr_scheduler, epoch, seg_accu, binary_loss,
                      loss_1, loss_2, loss_3, loss_4)