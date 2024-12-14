from utils import *
from IMD_dataloader import *
from custom_loss import IsolatingLossFunction, load_center_radius
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_cfg_defaults
from models.SALocation import SALLoc

from sklearn import metrics
import torch
import torch.nn as nn
import argparse
import numpy as np

device = torch.device('cuda:0')
device_ids = [0]

def config(args):
    args.crop_size = [256, 256]
    cuda_list = [0]
    global device
    device = torch.device('cuda:0')
    global device_ids
    device_ids = device_ids_return(cuda_list)

    args.save_dir    =  str(args.learning_rate)
    FENet_dir, SegNet_dir = args.save_dir+'/HRNet', args.save_dir+'/SALocal'
    FENet_cfg = get_cfg_defaults()
    FENet  = get_seg_model(FENet_cfg).to(device)
    SegNet = SALLoc().to(device)

    FENet  = nn.DataParallel(FENet, device_ids=device_ids)
    SegNet = nn.DataParallel(SegNet, device_ids=device_ids)

    make_folder(args.save_dir)
    make_folder(FENet_dir)
    make_folder(SegNet_dir)
    writer = SummaryWriter(f'tb_logs/{args.save_dir}')

    return args, writer, FENet, SegNet, FENet_dir, SegNet_dir

def restore_weight(args, FENet, SegNet, FENet_dir, SegNet_dir):
    params      = list(FENet.parameters()) + list(SegNet.parameters()) 
    optimizer   = torch.optim.Adam(params, lr=args.learning_rate)
    initial_epoch = findLastCheckpoint(save_dir=SegNet_dir)

    FENet  = restore_weight_helper(FENet,  FENet_dir,  initial_epoch)
    SegNet = restore_weight_helper(SegNet, SegNet_dir, initial_epoch)
    optimizer  = restore_optimizer(optimizer, SegNet_dir, initial_epoch)

    return optimizer, initial_epoch

def save_weight(FENet, SegNet, FENet_dir, SegNet_dir, optimizer, epoch):
    # Save checkpoint
    FENet_checkpoint = {'model': FENet.state_dict(),
                        'optimizer': optimizer.state_dict()}
    torch.save(FENet_checkpoint, '{0}/{1}.pth'.format(FENet_dir, epoch + 1))

    SegNet_checkpoint = {'model': SegNet.state_dict(),
                         'optimizer': optimizer.state_dict()}
    torch.save(SegNet_checkpoint, '{0}/{1}.pth'.format(SegNet_dir, epoch + 1))

def validation(
            args, FENet, SegNet, LOSS_MAP, tb_writer, 
            iter_num=None, 
            save_tag=False, 
            localization=True
            ):
    val_data_loader = infer_dataset_loader_init(args)
    val_num_per_epoch = len(val_data_loader)
    F1_lst, auc_lst = [], []

    with torch.no_grad():
        FENet.eval()
        SegNet.eval()
        for step, val_data in enumerate(tqdm(val_data_loader)):
            image, masks, cls0, cls1, cls2, cls3, image_names = val_data
            mask1, mask2, mask3, mask4 = masks
            mask1 = torch.squeeze(mask1, axis=1)
            image = image.to(device)
            mask1, mask2, mask3, mask4 = mask1.to(device), mask2.to(device), mask3.to(device), mask4.to(device)
            cls0, cls1, cls2, cls3 = cls0.to(device), cls1.to(device), cls2.to(device), cls3.to(device)

            # model 
            output = FENet(image)
            mask1_fea, mask_binary, out0, out1, out2, out3 = SegNet(output, image)
            if args.loss_type == 'D':
                loss_map, loss_manip, loss_nat = LOSS_MAP(mask1_fea, mask1)
                pred_mask = LOSS_MAP.dis_curBatch.squeeze(dim=1)
                pred_mask_score = LOSS_MAP.dist.squeeze(dim=1)
            elif args.loss_type == 'B':
                pred_mask_score = mask_binary
                pred_mask = torch.zeros_like(mask_binary)
                pred_mask[mask_binary > 0.5] = 1
                pred_mask[mask_binary <= 0.5] = 0
            viz_log(args, mask1, pred_mask, image, iter_num, step, mode='val')
            mask = torch.unsqueeze(mask1, axis=1)
            for img_idx, cur_img_name in enumerate(image_names):

                mask_ = torch.unsqueeze(mask[img_idx,0], 0)
                pred_mask_ = torch.unsqueeze(pred_mask[img_idx], 0)
                pred_mask_score_ = torch.unsqueeze(pred_mask_score[img_idx], 0)

                mask_ = mask_.cpu().clone().cpu().numpy().reshape(-1)
                pred_mask_ = pred_mask_.cpu().clone().cpu().numpy().reshape(-1)
                pred_mask_score_ = pred_mask_score_.cpu().clone().cpu().numpy().reshape(-1)

                F1_a  = metrics.f1_score(mask_, pred_mask_, average='macro')
                auc_a = metrics.roc_auc_score(mask_, pred_mask_score_)

                pred_mask_[np.where(pred_mask_ == 0)] = 1
                pred_mask_[np.where(pred_mask_ == 1)] = 0

                F1_b  = metrics.f1_score(mask_, pred_mask_, average='macro')
                if F1_a > F1_b:
                    F1 = F1_a
                else:
                    F1 = F1_b
                F1_lst.append(F1)
                AUC_score = auc_a if auc_a > 0.5 else 1-auc_a
                auc_lst.append(AUC_score)

    FENet.train()
    SegNet.train()
    print("...computing the pixel-wise scores/metrics here...")
    print(f"the scr_auc is: {np.mean(auc_lst):.3f}.")
    print(f"the macro is: {np.mean(F1_lst):.3f}")

def Inference(
            args, FENet, SegNet, LOSS_MAP, tb_writer, 
            iter_num=None, 
            save_tag=False, 
            localization=True
            ):
    for val_tag in [0,1,2,3,4]:
        val_data_loader, data_label = eval_dataset_loader_init(args, val_tag)
        print(f"working on the dataset: {data_label}.")
        F1_lst, auc_lst = [], []
        with torch.no_grad():
            FENet.eval()
            SegNet.eval()
            for step, val_data in enumerate(tqdm(val_data_loader)):
                image, mask, cls, image_names = val_data
                image, mask = image.to(device), mask.to(device)
                mask = torch.squeeze(mask, axis=1)
                # model 
                output = FENet(image)
                mask1_fea, mask_binary, out0, out1, out2, out3 = SegNet(output, image)
                if args.loss_type == 'D':
                    loss_map, loss_manip, loss_nat = LOSS_MAP(mask1_fea, mask)
                    pred_mask = LOSS_MAP.dis_curBatch.squeeze(dim=1)
                    pred_mask_score = LOSS_MAP.dist.squeeze(dim=1)
                elif args.loss_type == 'B':
                    pred_mask_score = mask_binary
                    pred_mask = torch.zeros_like(mask_binary)
                    pred_mask[mask_binary > 0.5] = 1
                    pred_mask[mask_binary <= 0.5] = 0
                viz_log(args, mask, pred_mask, image, iter_num, f"{step}_{val_tag}", mode='eval')

                mask = torch.unsqueeze(mask, axis=1)
                for img_idx, cur_img_name in enumerate(image_names):

                    mask_ = torch.unsqueeze(mask[img_idx,0], 0)
                    pred_mask_ = torch.unsqueeze(pred_mask[img_idx], 0)
                    pred_mask_score_ = torch.unsqueeze(pred_mask_score[img_idx], 0)

                    mask_ = mask_.cpu().clone().cpu().numpy().reshape(-1)
                    pred_mask_ = pred_mask_.cpu().clone().cpu().numpy().reshape(-1)
                    pred_mask_score_ = pred_mask_score_.cpu().clone().cpu().numpy().reshape(-1)

                    F1_a  = metrics.f1_score(mask_, pred_mask_, average='macro')
                    auc_a = metrics.roc_auc_score(mask_, pred_mask_score_)

                    pred_mask_[np.where(pred_mask_ == 0)] = 1
                    pred_mask_[np.where(pred_mask_ == 1)] = 0

                    F1_b  = metrics.f1_score(mask_, pred_mask_, average='macro')
                    if F1_a > F1_b:
                        F1 = F1_a
                    else:
                        F1 = F1_b
                    F1_lst.append(F1)
                    AUC_score = auc_a if auc_a > 0.5 else 1-auc_a
                    auc_lst.append(AUC_score)

        FENet.train()
        SegNet.train()
        print("F1: ", np.mean(F1_lst))
        print("AUC: ", np.mean(auc_lst))

def main(args):

    args, writer, FENet, SegNet, FENet_dir, SegNet_dir = config(args)

    train_data_loader = train_dataset_loader_init(args)
    train_num_per_epoch = int(args.train_num/args.train_bs)
    optimizer, lr_scheduler = setup_optimizer(args, SegNet, FENet)
    optimizer, initial_iter = restore_weight(args, FENet, SegNet, FENet_dir, SegNet_dir)

    center, radius = load_center_radius(args, FENet, SegNet, train_data_loader)
    CE_loss  = nn.CrossEntropyLoss().to(device)
    BCE_loss = nn.BCELoss(reduction='none').to(device)
    if args.loss_type == 'B':
        LOSS_MAP = None
    elif args.loss_type == 'D':
        LOSS_MAP = IsolatingLossFunction(center,radius).to(device)

    for epoch in range(0, args.num_epochs):
        seg_total, seg_correct, seg_loss_sum = [0]*3
        map_loss_sum, mani_lss_sum, natu_lss_sum, binary_map_loss_sum = [0]*4
        loss_1_sum, loss_2_sum, loss_3_sum, loss_4_sum = [0]*4

        for step, train_data in enumerate(train_data_loader):
            iter_num = epoch * train_num_per_epoch + step
            image, masks, cls0, cls1, cls2, cls3 = train_data
            mask1, mask2, mask3, mask4 = masks
            image = image.to(device)
            mask1 = mask1.to(device)
            cls0, cls1, cls2, cls3 = cls0.to(device), cls1.to(device), cls2.to(device), cls3.to(device)
            mask1, mask1_balance = class_weight(mask1, 1)

            # model 
            output = FENet(image)
            mask1_fea, mask_binary, out0, out1, out2, out3 = SegNet(output, image)
            # objective
            loss_4 = CE_loss(out3, cls3)
            forgery_cls = ~(cls0.eq(0))
            if np.sum(forgery_cls.cpu().numpy()) != 0:
                loss_1 = CE_loss(out0[forgery_cls,:], cls0[forgery_cls])
                loss_2 = CE_loss(out1[forgery_cls,:], cls1[forgery_cls])
                loss_3 = CE_loss(out2[forgery_cls,:], cls2[forgery_cls])
            else:
                loss_1 = torch.tensor(0.0, requires_grad=True).to(device)
                loss_2 = torch.tensor(0.0, requires_grad=True).to(device)
                loss_3 = torch.tensor(0.0, requires_grad=True).to(device)

            loss_binary_map = torch.mean(BCE_loss(mask_binary, mask1.to(torch.float)) * mask1_balance)
            if args.loss_type == 'D':
                loss, loss_manip, loss_nat = LOSS_MAP(mask1_fea, mask1)
            elif args.loss_type == 'B':
                loss = loss_binary_map
                loss_manip = loss_binary_map
                loss_nat = loss_binary_map
            loss_total = composite_obj(args, loss, loss_1, loss_2, loss_3, loss_4, loss_binary_map)
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            if args.loss_type == 'D':
                pred_mask1 = LOSS_MAP.dis_curBatch.squeeze(dim=1)
            elif args.loss_type == 'B':
                pred_mask1 = torch.zeros_like(mask_binary)
                pred_mask1[mask_binary > 0.5] = 1
                pred_mask1[mask_binary <= 0.5] = 0
            seg_correct += (pred_mask1 == mask1).sum().item()
            seg_total   += int(torch.ones_like(mask1).sum().item())
            map_loss_sum += loss.item()
            mani_lss_sum += loss_manip.item()
            natu_lss_sum += loss_nat.item()
            binary_map_loss_sum += loss_binary_map.item()
            loss_1_sum += loss_1.item()
            loss_2_sum += loss_2.item()
            loss_3_sum += loss_3.item()
            loss_4_sum += loss_4.item()

            if step % args.dis_step == 0:
                train_log_dump(
                            args, seg_correct, seg_total, binary_map_loss_sum, loss_1_sum, loss_2_sum,
                            loss_3_sum, loss_4_sum, epoch, step, writer, iter_num,
                            lr_scheduler
                            )
                schedule_step_loss = composite_obj_step(args, loss_4_sum, map_loss_sum)
                lr_scheduler.step(schedule_step_loss)
                seg_total, seg_correct, seg_loss_sum = [0]*3
                loss_1_sum, loss_2_sum, loss_3_sum, loss_4_sum = [0]*4
                map_loss_sum, mani_lss_sum, natu_lss_sum, binary_map_loss_sum = [0]*4
                viz_log(args, mask1, pred_mask1, image, iter_num, step, mode='train')

            if (iter_num+1) % args.val_step == 0:
                
                validation(
                        args, 
                        FENet, 
                        SegNet, 
                        LOSS_MAP,
                        tb_writer=writer, 
                        iter_num=iter_num, 
                        save_tag=True, 
                        localization=True
                        )
                
                if(iter_num+1) % args.Inf_step == 0:
                    Inference(
                            args,
                            FENet,
                            SegNet,
                            LOSS_MAP,
                            tb_writer=writer,
                            iter_num=iter_num,
                            save_tag=True,
                            localization=True
                            )
                print(f"...save the iteration number: {iter_num}.")
                save_weight(FENet, SegNet, FENet_dir, SegNet_dir, optimizer, iter_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--lr_backbone', type=float, default=0.9)
    parser.add_argument('--patience', type=int, default=70)
    parser.add_argument('--step_factor', type=float, default=0.95)
    parser.add_argument('--dis_step', type=int, default=50)
    parser.add_argument('--val_step', type=int, default=500)
    parser.add_argument('--Inf_step', type=int, default=1000)
    parser.add_argument('--train_bs', type=int, default=12)
    parser.add_argument('--val_bs', type=int, default=12)
    parser.add_argument('--val_num', type=int, default=15000)
    parser.add_argument('--train_num', type=int, default=150000)
    parser.add_argument('--ablation', type=str, default='local', choices=['base', 'fg', 'local', 'full'])
    parser.add_argument('--path', type=str, default="", help='deprecated')
    parser.add_argument('--loss_type', type=str, default='B', choices=['B', 'D'])
    args = parser.parse_args()
    main(args)
