import argparse
import os
from datetime import datetime
from distutils.dir_util import copy_tree
# from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from itertools import cycle
import random

# import torch.backends.cudnn as cudnn
# import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.modules.loss import CrossEntropyLoss
from utilities.dataloaders import*
# from new_dataloaders import*
# from loss import loss_sup, loss_adversarial_1, loss_adversarial_2, make_Dis_label, gt_label, loss_diff
from utilities.metrics import*
from utilities.losses_1 import*
from utilities.losses_2 import*
from utilities.pytorch_losses import dice_loss
from utilities.ramps import sigmoid_rampup
from CPS_model import model1, model2
from utilities.utilities import get_logger, create_dir 
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # specify which GPU(s) to be used
seed = 1337
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"  # specify which GPU(s) to be used

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30250, help='maximum epoch number to train')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')

parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

args = parser.parse_args()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # specify the GPU id's, GPU id's start from 0.


epochs = 800
# batchsize = 16 
# CE = torch.nn.BCELoss()
# criterion_1 = torch.nn.BCELoss()
num_classes = args.num_classes


# kl_distance = nn.KLDivLoss(reduction='none') #KL_loss for consistency training
ce_loss = CrossEntropyLoss()
# dice_loss = 1 - mDice(pred_mask, mask)
base_lr = args.base_lr
max_iterations = args.max_iterations

sim_loss = feature_sim()

iter_per_epoch = 60

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)



class Network(object):
    def __init__(self):
        self.patience_1 = 0
        self.patience_2 = 0
        self.best_dice_coeff_1 = False
        self.best_dice_coeff_2 = False
        self.model1 = model1
        self.model2 = model2
        # self.model3 = model3
        self._init_logger()

    # def _init_configure(self):
    #     with open('configs/config.yml') as fp:
    #         self.cfg = yaml.safe_load(fp)

    def _init_logger(self):

        log_dir = '/.../model_weights/NEU_seg/'

        self.logger = get_logger(log_dir)
        print('RUNDIR: {}'.format(log_dir))

        self.save_path = log_dir

        self.save_tbx_log = self.save_path + '/tbx_log'
        self.writer = SummaryWriter(self.save_tbx_log)

    def run(self):
        self.model1.to(device)
        self.model2.to(device)
        optimizer_1 = torch.optim.Adam(self.model1.parameters(), lr=base_lr)
        optimizer_2 = torch.optim.Adam(self.model2.parameters(), lr=base_lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=20, verbose=True)
        scheduler_1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_1, mode="max", min_lr = 0.0000001, patience=50, verbose=True)
        scheduler_2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_2, mode="max", min_lr = 0.0000001, patience=50, verbose=True)
      
        
        self.logger.info(
            "train_loader {} unlabeled_loader {} val_loader {} test_loader {}".format(len(train_loader),
                                                                                      len(unlabeled_loader),
                                                                                      len(val_loader),
                                                                                      len(test_loader)))
        print("Training process started!")
        print("===============================================================================================")

        # model1.train()
        iter_num = 0
       
        for epoch in range(1, epochs):

            running_ce_loss_1 = 0.0
            running_dice_loss_1 = 0.0
            running_ce_loss_2 = 0.0
            running_dice_loss_2 = 0.0
            running_train_iou_1 = 0.0
            running_train_dice_1 = 0.0
            running_train_iou_2 = 0.0
            running_train_dice_2 = 0.0
            running_labeled_cps = 0.0
            # running_consistency_loss = 0.0
            # running_dice_loss_3 = 0.0
            
            running_train_loss = 0.0
            running_cps_loss = 0.0
            running_sim_loss = 0.0
            
            running_val_loss = 0.0
            running_dice_loss_val_1 = 0.0
            running_ce_loss_val_1 = 0.0
            running_dice_loss_val_2 = 0.0
            running_ce_loss_val_2 = 0.0

            running_val_iou_1 = 0.0
            running_val_dice_1 = 0.0
            running_val_accuracy_1 = 0.0

            running_val_iou_2 = 0.0
            running_val_dice_2 = 0.0
            running_val_accuracy_2 = 0.0

                        
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            # optimizer_3.zero_grad()
            
            self.model1.train()
            self.model2.train()
            # self.model3.train()

            semi_dataloader = iter(zip(cycle(train_loader), unlabeled_loader))
                    
            for iteration in range (1, iter_per_epoch): #(zip(train_loader, unlabeled_train_loader)):
                
                data = next(semi_dataloader)
                
                (inputs_S1, labels_S1), (inputs_U, labels_U) = data #data[0][0], data[0][1]


                inputs_S1, labels_S1 = Variable(inputs_S1), Variable(labels_S1)
                inputs_S1, labels_S1 = inputs_S1.to(device), labels_S1.to(device)

                inputs_U, labels_U = Variable(inputs_U), Variable(labels_U)
                inputs_U, labels_U = inputs_U.to(device), labels_U.to(device)

                self.model1.train()
                self.model2.train()
                # self.model3.train()

                # Train Model 1
                #Labeled samples output
                x4_1, _, _, _, f4_1, outputs_1, _, _, _ = self.model1(inputs_S1)
                x4_2, _, _, _, f4_2, outputs_2, _, _, _ = self.model2(inputs_S1)
                # outputs_2 = self.model2(inputs_S1) #Unet res18
                # x4, f1, f2, f3, f4, dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg = self.model3(inputs_S1)
                
                outputs_1_soft = torch.softmax(outputs_1, dim=1)
                # outputs_1_aux_soft = torch.softmax(outputs_aux_1, dim=1)
                outputs_2_soft = torch.softmax(outputs_2, dim=1)
                # outputs_2_aux_soft = torch.softmax(outputs_aux_2, dim=1)
                
                #Unlabeled samples output
                x4_1_un, _, _, _, f4_1_un, un_outputs_1, _, _, _ = self.model1(inputs_U)
                x4_2_un, _, _, _, f4_2_un, un_outputs_2, _, _, _ = self.model2(inputs_U)
                #Softmax output
                un_outputs_soft_1 = torch.softmax(un_outputs_1, dim=1)
                # un_outputs_aux_soft_1 = torch.softmax(un_outputs_aux_1, dim=1)
                un_outputs_soft_2 = torch.softmax(un_outputs_2, dim=1)
                # un_outputs_aux_soft_2 = torch.softmax(un_outputs_aux_2, dim=1)
                
                #CE_loss
                loss_ce_1 = ce_loss(outputs_1, labels_S1.long())
                loss_ce_2 = ce_loss(outputs_2, labels_S1.long())
                
                #Dice_loss
                loss_dice_1 = dice_loss(labels_S1.unsqueeze(1), outputs_1)
                loss_dice_2 = dice_loss(labels_S1.unsqueeze(1), outputs_2)

                model1_sup_loss =0.5*(loss_ce_1 + loss_dice_1) 
                model2_sup_loss =0.5*(loss_ce_2 + loss_dice_2) 
                sup_loss = model1_sup_loss + model2_sup_loss

                
                #Pseudo_labels for labeles samples
                lbl_pseudo_1 = torch.argmax(outputs_1_soft.detach(), dim=1, keepdim=False)
                lbl_pseudo_2 = torch.argmax(outputs_2_soft.detach(), dim=1, keepdim=False)

                psd_sup1 = 0.5*ce_loss(outputs_1, lbl_pseudo_2) + 0.5*dice_loss(lbl_pseudo_2.unsqueeze(1), outputs_1)
                psd_sup2 = 0.5*ce_loss(outputs_2, lbl_pseudo_1) + 0.5*dice_loss(lbl_pseudo_1.unsqueeze(1), outputs_2)

                labeled_cps = psd_sup1 + psd_sup2
                
                #Pseudo-labels

                pseudo_outputs1 = torch.argmax(un_outputs_soft_1.detach(), dim=1, keepdim=False)
                pseudo_outputs2 = torch.argmax(un_outputs_soft_2.detach(), dim=1, keepdim=False)

                pseudo_supervision1 = 0.5*ce_loss(un_outputs_1, pseudo_outputs2) + 0.5*dice_loss(pseudo_outputs2.unsqueeze(1), un_outputs_1)
                pseudo_supervision2 = 0.5*ce_loss(un_outputs_2, pseudo_outputs1) + 0.5*dice_loss(pseudo_outputs1.unsqueeze(1), un_outputs_2)
                

                cps_loss = pseudo_supervision1 +  pseudo_supervision2 

                consistency_weight = get_current_consistency_weight(iter_num // 150) #Consistency weight multipliers

                
                loss = sup_loss + consistency_weight * cps_loss + consistency_weight*labeled_cps  
                
                optimizer_1.zero_grad()
                optimizer_2.zero_grad()
                
                loss.backward()

                # if (i + 1 ) % self.accumulation_steps == 0:
                #     optimizer.step()
                #     optimizer.zero_grad()
                optimizer_1.step()
                optimizer_2.step()
                # scheduler_1.step()
                # scheduler_2.step()
                # optimizer.zero_grad()
                running_train_loss += loss.item()
                running_ce_loss_1 += loss_ce_1.item()
                running_dice_loss_1 += loss_dice_1.item()

                running_ce_loss_2 += loss_ce_2.item()
                running_dice_loss_2 += loss_dice_2.item()
                running_cps_loss += cps_loss.item()
                running_labeled_cps += labeled_cps.item()
                
                running_train_iou_1 += mIoU(outputs_1, labels_S1)
                running_train_dice_1 += mDice(outputs_1, labels_S1)

                running_train_iou_2 += mIoU(outputs_2, labels_S1)
                running_train_dice_2 += mDice(outputs_2, labels_S1)

                
                for param_group in optimizer_1.param_groups:
                    lr_1 = param_group['lr'] #For plotting the learning rate change during the training process

                
                for param_group in optimizer_2.param_groups:
                    lr_2 = param_group['lr'] #For plotting the learning rate change during the training process# lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                

                # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                # for param_group in optimizer_2.param_groups:
                #     param_group['lr'] = lr_


                
                iter_num = iter_num + 1
            
            epoch_train_dice_1 = ( running_train_dice_1) / (iter_per_epoch)
            epoch_train_iou_1 = ( running_train_iou_1) / (iter_per_epoch)
            epoch_train_iou_2 = ( running_train_iou_2) / (iter_per_epoch)
            epoch_train_dice_2 = ( running_train_dice_2) / (iter_per_epoch)

            epoch_loss = (running_train_loss) / (iter_per_epoch)
            epoch_ce_loss_1 = (running_ce_loss_1) / (iter_per_epoch)
            epoch_dice_loss_1 = (running_dice_loss_1) / (iter_per_epoch)
            epoch_ce_loss_2 = (running_ce_loss_2) / (iter_per_epoch)
            epoch_dice_loss_2 = (running_dice_loss_2) / (iter_per_epoch)
            epoch_cps_loss = (running_cps_loss) / (iter_per_epoch)
            epoch_labeled_cps_loss = (running_labeled_cps) / (iter_per_epoch)

            self.logger.info('{} Epoch [{:03d}/{:03d}], total_loss : {:.4f}'.
                             format(datetime.now(), epoch, epochs, epoch_loss))

            self.logger.info('Train loss: {}'.format(epoch_loss))
            self.writer.add_scalar('Train/Loss', epoch_loss, epoch)

            self.logger.info('Train IoU-1: {}'.format(epoch_train_iou_1))
            self.writer.add_scalar('Train/IoU-1', epoch_train_iou_1, epoch)
            self.logger.info('Train Dice-1: {}'.format(epoch_train_dice_1))
            self.writer.add_scalar('Train/Dice-1', epoch_train_dice_1, epoch)

            self.logger.info('Train IoU-2: {}'.format(epoch_train_iou_2))
            self.writer.add_scalar('Train/IoU-2', epoch_train_iou_2, epoch)
            self.logger.info('Train Dice-2: {}'.format(epoch_train_dice_2))
            self.writer.add_scalar('Train/Dice-2', epoch_train_dice_2, epoch)
            
            self.logger.info('Train ce-loss-1: {}'.format(epoch_ce_loss_1))
            self.writer.add_scalar('Train/CE-Loss-1', epoch_ce_loss_1, epoch)
            self.logger.info('Train dice-loss-1: {}'.format(epoch_dice_loss_1))
            self.writer.add_scalar('Train/Dice-Loss-1', epoch_dice_loss_1, epoch)

            self.logger.info('Train ce-loss-2: {}'.format(epoch_ce_loss_2))
            self.writer.add_scalar('Train/CE-Loss-2', epoch_ce_loss_2, epoch)
            self.logger.info('Train dice-loss-2: {}'.format(epoch_dice_loss_2))
            self.writer.add_scalar('Train/Dice-Loss-2', epoch_dice_loss_2, epoch)

            # self.logger.info('Train dice-loss: {}'.format(epoch_dice_loss))
            # self.writer.add_scalar('Train/Dice-Loss', epoch_dice_loss, epoch)

            self.logger.info('Train CPS-loss: {}'.format(epoch_cps_loss))
            self.writer.add_scalar('Train/CPS-Loss', epoch_cps_loss, epoch)
            self.logger.info('Train labeled-CPS-loss: {}'.format(epoch_labeled_cps_loss))
            self.writer.add_scalar('Train/labeled-CPS-Loss', epoch_labeled_cps_loss, epoch)

            # self.writer.add_scalar('info/lr', lr_, epoch)
            self.writer.add_scalar('info/lr1', lr_1, epoch)
            self.writer.add_scalar('info/lr2', lr_2, epoch)
            self.writer.add_scalar('info/consis_weight', consistency_weight, epoch)
            torch.cuda.empty_cache()

            self.model1.eval()
            self.model2.eval()
            for i, pack in enumerate(val_loader, start=1):
                with torch.no_grad():
                    images, gts = pack
                    # images = Variable(images)
                    # gts = Variable(gts)
                    images = images.to(device)
                    gts = gts.to(device)
                    
                    _, _, _, _, _, pred_1, _, _, _= self.model1(images)
                    _, _, _, _, _, pred_2, _, _, _= self.model2(images)
                    # pred_2 = self.model2(images)
                    # Prediction_1_soft = torch.softmax(prediction_1, dim=1)

                        

                # dice_coe_1 = dice_coef(prediction_1, gts)
                loss_ce_1 = ce_loss(pred_1, gts.long())
                loss_dice_1 = 1 - mDice(pred_1, gts)

                loss_ce_2 = ce_loss(pred_2, gts.long())
                loss_dice_2 = 1 - mDice(pred_2, gts)
                
                # loss_ce = loss_ce_1 + loss_ce_2
                # loss_dice = loss_dice_1 + loss_dice_2

                val_loss = 0.5 * (loss_dice_1 + loss_ce_1) + 0.5 * (loss_dice_2 + loss_ce_2)

                running_val_loss += val_loss.item()
                running_dice_loss_val_1 += loss_dice_1.item()
                running_ce_loss_val_1 += loss_ce_1.item()
                running_dice_loss_val_2 += loss_dice_2.item()
                running_ce_loss_val_2 += loss_ce_2.item()



                running_val_iou_1 += mIoU(pred_1, gts)
                running_val_dice_1 += mDice(pred_1, gts)
                running_val_accuracy_1 += pixel_accuracy(pred_1, gts)

                running_val_iou_2 += mIoU(pred_2, gts)
                running_val_dice_2 += mDice(pred_2, gts)
                running_val_accuracy_2 += pixel_accuracy(pred_2, gts)

                # running_val_iou_2 += mIoU(prediction_2, gts)
                # running_val_accuracy_2 += pixel_accuracy(prediction_2, gts)
                # running_val_dice_2 += mDice(prediction_2, gts)
                
                 
            epoch_loss_val = running_val_loss / len(val_loader)
            epoch_val_dice_loss_1 = running_dice_loss_val_1 / len(val_loader)
            epoch_val_ce_loss_1 = running_ce_loss_val_1 / len(val_loader)
            epoch_val_dice_loss_2 = running_dice_loss_val_2 / len(val_loader)
            epoch_val_ce_loss_2 = running_ce_loss_val_2 / len(val_loader)


            epoch_dice_val_1 = running_val_dice_1 / len(val_loader)
            epoch_iou_val_1 = running_val_iou_1 / len(val_loader)
            epoch_accuracy_val_1 = running_val_accuracy_1 / len(val_loader)

            epoch_dice_val_2 = running_val_dice_2 / len(val_loader)
            epoch_iou_val_2 = running_val_iou_2 / len(val_loader)
            epoch_accuracy_val_2 = running_val_accuracy_2 / len(val_loader)
            scheduler_1.step(epoch_dice_val_1)
            scheduler_2.step(epoch_dice_val_2)
            
            self.logger.info('Val loss: {}'.format(epoch_loss_val))
            self.writer.add_scalar('Val/loss', epoch_loss_val, epoch)

            #model-1 perfromance
            self.logger.info('Val dice_loss_1 : {}'.format(epoch_val_dice_loss_1))
            self.writer.add_scalar('Val/Dice-loss_1', epoch_val_dice_loss_1, epoch)
            self.logger.info('Val ce_loss_1 : {}'.format(epoch_val_ce_loss_1))
            self.writer.add_scalar('Val/ce-loss_1', epoch_val_ce_loss_1, epoch)

            self.logger.info('Val dice_loss_2 : {}'.format(epoch_val_dice_loss_2))
            self.writer.add_scalar('Val/Dice-loss_2', epoch_val_dice_loss_2, epoch)
            self.logger.info('Val ce_loss_2 : {}'.format(epoch_val_ce_loss_2))
            self.writer.add_scalar('Val/ce-loss_2', epoch_val_ce_loss_2, epoch)

            self.logger.info('Val dice_1 : {}'.format(epoch_dice_val_1))
            self.writer.add_scalar('Val/DSC-1', epoch_dice_val_1, epoch)

            self.logger.info('Val IoU_1 : {}'.format(epoch_iou_val_1))
            self.writer.add_scalar('Val/IoU-1', epoch_iou_val_1, epoch)

            self.logger.info('Val Accuracy_1 : {}'.format(epoch_accuracy_val_1))
            self.writer.add_scalar('Val/Accuracy-1', epoch_accuracy_val_1, epoch)

            #model-2 training

            self.logger.info('Val dice_2 : {}'.format(epoch_dice_val_2))
            self.writer.add_scalar('Val/DSC-2', epoch_dice_val_2, epoch)

            self.logger.info('Val IoU_2 : {}'.format(epoch_iou_val_2))
            self.writer.add_scalar('Val/IoU-2', epoch_iou_val_2, epoch)

            self.logger.info('Val Accuracy_2 : {}'.format(epoch_accuracy_val_2))
            self.writer.add_scalar('Val/Accuracy-2', epoch_accuracy_val_2, epoch)


            
            mdice_coeff_1 =  epoch_dice_val_1
            mdice_coeff_2 =  epoch_dice_val_2
            # mval_loss_1 = epoch_val_loss

            if self.best_dice_coeff_1 < mdice_coeff_1:
                self.best_dice_coeff_1 = mdice_coeff_1
                self.save_best_model_1 = True
                self.patience_1 = 0
            else:
                self.save_best_model_1 = False
                self.patience_1 += 1


            if self.best_dice_coeff_2 < mdice_coeff_2:
                self.best_dice_coeff_2 = mdice_coeff_2
                self.save_best_model_2 = True
                self.patience_2 = 0
            else:
                self.save_best_model_2 = False
                self.patience_2 += 1            
            
            Checkpoints_Path = self.save_path + '/Checkpoints'

            if not os.path.exists(Checkpoints_Path):
                os.makedirs(Checkpoints_Path)

            if self.save_best_model_1:
                state_1 = {
                "epoch": epoch,
                "best_dice_1": self.best_dice_coeff_1,
                "state_dict": self.model1.state_dict(),
                "optimizer": optimizer_1.state_dict(),
                }
                # state["best_loss"] = self.best_loss
                torch.save(state_1, Checkpoints_Path + '/CPS_10p_1.pth')

            
            if self.save_best_model_2:
                state_2 = {
                "epoch": epoch,
                "best_dice_2": self.best_dice_coeff_2,
                "state_dict": self.model2.state_dict(),
                "optimizer": optimizer_2.state_dict(),
                }
                # state["best_loss"] = self.best_loss
                torch.save(state_2, Checkpoints_Path + '/CPS_10p_2.pth')
  
 
            
            
             
            self.logger.info(
                'current best dice coef: model-1: {}, model-2: {}'.format(self.best_dice_coeff_1, self.best_dice_coeff_2))

            self.logger.info('current patience: m1: {}, m2: {}'.format(self.patience_1, self.patience_2))
            print('Current consistency weight:', consistency_weight)
            print('Current iteration:', iter_num)
            print('================================================================================================')
            print('================================================================================================')

            #if iter_num




if __name__ == '__main__':
    train_network = Network()
    train_network.run()