import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # specify which GPU(s) to be used
from datetime import datetime
from distutils.dir_util import copy_tree
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from itertools import cycle

# import torch.backends.cudnn as cudnn
# import yaml
from tensorboardX import SummaryWriter

from torch.autograd import Variable
from torch.nn.modules.loss import CrossEntropyLoss
from UCC_dataloaders import*
from cross_cutmix import generate_mix_data, generate_crossmix_data

from utilities.metrics import*
from utilities.losses_1 import*
from utilities.losses_2 import*
from utilities.pytorch_losses import dice_loss
from utilities.ramps import sigmoid_rampup
from UCC_model import model
from utilities.utilities import get_logger, create_dir

import os
seed = 1337
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,
                    default='unet_ccps', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=20000, help='maximum epoch number to train')
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # specify the GPU id's, GPU id's start from 0.


epochs = 800
# batchsize = 16 
# CE = torch.nn.BCELoss()
# criterion_1 = torch.nn.BCELoss()
num_classes = args.num_classes


kl_distance = nn.KLDivLoss(reduction='none') #KL_loss for consistency training
log_sm = torch.nn.LogSoftmax(dim = 1) #For computing the KL distance
ce_loss = CrossEntropyLoss()
base_lr = args.base_lr
max_iterations = args.max_iterations
iter_per_epoch = 60

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)



class Network(object):
    def __init__(self):
        self.patience = 0
        self.best_dice_coeff_1 = False
        self.model = model
        self._init_logger()

    def _init_logger(self):

        log_dir = '.../model_weights'

        self.logger = get_logger(log_dir)
        print('RUNDIR: {}'.format(log_dir))

        self.save_path = log_dir
        self.save_tbx_log = self.save_path + '/tbx_log'
        self.writer = SummaryWriter(self.save_tbx_log)

    def run(self):
        self.model.to(device)
        optimizer_1 = torch.optim.Adam(self.model.parameters(), lr=base_lr)
        scheduler_1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_1, mode="max", min_lr = 0.00000001, patience=50, verbose=True)
      
        
        self.logger.info(
            "train_loader {} unlabeled_loader {} val_loader {} test_loader {} ".format(len(train_loader),
                                                                       len(unlabeled_loader),
                                                                       len(val_loader),
                                                                       len(test_loader)))
        print("Training process started!")
        print("===============================================================================================")

        # model1.train()
        iter_num = 0
       
        for epoch in range(1, epochs):

            running_train_ce_loss = 0.0
            running_train_dice_loss = 0.0
            running_train_loss = 0.0
            running_train_iou_1 = 0.0
            running_train_iou_2= 0.0
            running_train_dice_1 = 0.0
            running_train_dice_2 = 0.0
            running_train_ps_loss = 0.0
            running_consistency_loss = 0.0

            running_val_ce_loss = 0.0
            running_val_dice_loss = 0.0
            running_val_loss = 0.0                               
            running_val_iou_1 = 0.0
            running_val_dice_1 = 0.0
            running_val_accuracy_1 = 0.0

                        
            optimizer_1.zero_grad()
            # optimizer_2.zero_grad()
            
            self.model.train()
            # self.model_2.train()

            semi_dataloader = iter(zip(cycle(train_loader), unlabeled_loader))
                    
            for iteration in range (1, iter_per_epoch): #(zip(train_loader, unlabeled_train_loader)):
                
                data = next(semi_dataloader)
                
                (inputs_lbl, labels_lbl), (inputs_U1, inputs_U2, labels_U1, labels_U2) = data #data[0][0], data[0][1]

                #-----------------------------------------------------------------
                #Applying the cross-cutmix augumentation to the unlabeled samples
                #-----------------------------------------------------------------
                inputs_l, labels_l = generate_mix_data(inputs_lbl, labels_lbl)
                new_inputs_U1, new_inputs_U2 = generate_crossmix_data(inputs_l, inputs_U1, inputs_U2) #Cut and mix labeled with the unlabeled
                
                inputs_l, labels_l = Variable(inputs_l), Variable(labels_l)
                inputs_l, labels_l = inputs_l.to(device), labels_l.to(device)

                new_inputs_U1, labels_U1 = Variable(new_inputs_U1), Variable(labels_U1) #Weak branch
                new_inputs_U2, labels_U2 = Variable(new_inputs_U2), Variable(labels_U2) #Strong branch
                new_inputs_U1, labels_U1 = new_inputs_U1.to(device), labels_U1.to(device)
                new_inputs_U2, labels_U2 = new_inputs_U2.to(device), labels_U2.to(device)

                self.model.train()
                # self.model_2.train()

                # Train Model 1
                #Labeled samples output
                outputs1, outputs2 = self.model(inputs_l)
                outputs_1_soft = torch.softmax(outputs1, dim=1)
                outputs_2_soft = torch.softmax(outputs2, dim=1)
                
                # #Applying the cross-cutmix augumentation to the unlabeled samples
                # new_inputs_U1, new_inputs_U2 = generate_unsup_data(inputs_l, inputs_U1, inputs_U2) #Cut and mix labeled with the unlabeled
                
                #Unlabeled samples output on the weak branch
                un_outputs1_wk, un_outputs2_wk = self.model(new_inputs_U1)
                un_outputs1_wk_soft = torch.softmax(un_outputs1_wk, dim=1)
                un_outputs2_wk_soft = torch.softmax(un_outputs2_wk, dim=1)

                #Unlabeled samples output on the strong branch
                un_outputs1_st, un_outputs2_st = self.model(new_inputs_U2)
                un_outputs1_st_soft = torch.softmax(un_outputs1_st, dim=1)
                un_outputs2_st_soft = torch.softmax(un_outputs2_st, dim=1)
                          
                #CE_loss on labeled samples
                loss_ce_1 = ce_loss(outputs1, labels_l.long())
                loss_ce_2 = ce_loss(outputs2, labels_l.long())
                                
                #Dice_loss on labeled samples
                loss_dice_1 = dice_loss(labels_l.unsqueeze(1), outputs1)
                loss_dice_2 = dice_loss(labels_l.unsqueeze(1), outputs2)
                        
                           
                #-----------------------------------------------------------------
                #Total supervised losss
                #-----------------------------------------------------------------
                sup_loss_1 = 0.5*(loss_ce_1  + loss_dice_1)
                sup_loss_2 = 0.5*(loss_ce_2  + loss_dice_2)
                
                total_sup_loss_ce = (loss_ce_1 + loss_ce_2)/2  #for plotting epoch loss
                total_sup_loss_dice = (loss_dice_1 + loss_dice_2)/2  #for plotting epoch loss
                supervised_loss = sup_loss_1 + sup_loss_2  #for plotting epoch loss
                
                
                #Consistency loss on the predicion of each head
                # consistency_loss_1 = torch.mean((un_outputs1_wk_soft - un_outputs2_wk_soft) ** 2)
                # consistency_loss_2 = torch.mean((un_outputs1_st_soft - un_outputs2_st_soft) ** 2)
                # consistency_loss = (consistency_loss_1 + consistency_loss_2 ) / 2

                #-----------------------------------------------------------------
                # Compute uncertainty as predicion difference between head1 and head2
                #-----------------------------------------------------------------
                variance_1 = torch.sum(kl_distance(log_sm(un_outputs1_wk), un_outputs2_st_soft), dim=1)
                exp_variance_1 = torch.exp(-variance_1)

                variance_2 = torch.sum(kl_distance(log_sm(un_outputs1_st), un_outputs2_wk_soft), dim=1)
                exp_variance_2 = torch.exp(-variance_2)
                
                        
                #-----------------------------------------------------------------
                # Pseudo label for the unlabeled samples for each segmentation head 
                #-----------------------------------------------------------------

                pseudo_1 = torch.argmax(un_outputs2_wk_soft.detach(), dim=1, keepdim=False)# 
                pseudo_2 = torch.argmax(un_outputs1_wk_soft.detach(), dim=1, keepdim=False)# 
                
                
                #WITHOUT UNCERTAINTY
                # ps_1_wk = 0.5*(ce_loss(un_outputs1_st, pseudo_1) + dice_loss(pseudo_1.unsqueeze(1), un_outputs1_st))
                # ps_2_st = 0.5*(ce_loss(un_outputs2_st, pseudo_2) + dice_loss(pseudo_2.unsqueeze(1), un_outputs2_st))
                
                #WITH UNCERTAINTY
                ps_1_wk = torch.mean(0.5*(ce_loss(un_outputs1_st, pseudo_1) + dice_loss(pseudo_1.unsqueeze(1), un_outputs1_st))* exp_variance_1) + torch.mean(variance_1) #unlabeled samples psuedo-label
                ps_2_st = torch.mean(0.5*(ce_loss(un_outputs2_st, pseudo_2) + dice_loss(pseudo_2.unsqueeze(1), un_outputs2_st))* exp_variance_2) + torch.mean(variance_2)#unlabeled samples psuedo-label
                                         
               #OLY CE loss
                # ps_1_wk = torch.mean(ce_loss(un_outputs1_st, pseudo_1)* exp_variance_1) + torch.mean(variance_1) #unlabeled samples psuedo-label
                # ps_2_st = torch.mean(ce_loss(un_outputs2_st, pseudo_2)* exp_variance_2) + torch.mean(variance_2)#unlabeled samples psuedo-label
                
                ps_loss = ps_1_wk + ps_2_st

                #CONSISTENCY LOSS BETWEEN WEAK AND STRONG AUGMENTATION
              
                consistency_weight = get_current_consistency_weight(iter_num // 150) #Consistency weight multipliers
                
                loss = supervised_loss + consistency_weight*ps_loss #+ consistency_weight*consistency_loss
                
                
                optimizer_1.zero_grad()
                
                loss.backward()

                # if (i + 1 ) % self.accumulation_steps == 0:
                #     optimizer.step()
                #     optimizer.zero_grad()
                optimizer_1.step()
                # optimizer_2.step()
                # optimizer.zero_grad()
                running_train_loss += loss.item()
                running_train_ce_loss += total_sup_loss_ce.item()
                # running_consistency_loss += consistency_loss.item()
                running_train_dice_loss += total_sup_loss_dice.item()
                running_train_ps_loss += ps_loss.item()
                running_train_iou_1 += mIoU(outputs1, labels_l)
                running_train_iou_2 += mIoU(outputs2, labels_l)
                running_train_dice_1 += mDice(outputs1, labels_l)
                running_train_dice_2 += mDice(outputs2, labels_l)

                
                # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer_1.param_groups:
                    lr_ = param_group['lr']

                
                iter_num = iter_num + 1
            
            epoch_loss = (running_train_loss) / (iter_per_epoch)
            epoch_ce_loss = (running_train_ce_loss) / (iter_per_epoch)
            epoch_dice_loss = (running_train_dice_loss) / (iter_per_epoch)
            epoch_ps_loss = (running_train_ps_loss) / (iter_per_epoch)
            epoch_consistency_loss = (running_consistency_loss) / (iter_per_epoch)
            epoch_iou_1 = (running_train_iou_1) / (iter_per_epoch)
            epoch_iou_2 = (running_train_iou_2) / (iter_per_epoch)
            epoch_dice_1 = (running_train_dice_1) / (iter_per_epoch)
            epoch_dice_2 = (running_train_dice_2) / (iter_per_epoch)
 
            self.logger.info('{} Epoch [{:03d}/{:03d}], total_loss : {:.4f}'.
                             format(datetime.now(), epoch, epochs, epoch_loss))

            self.logger.info('Train loss: {}'.format(epoch_loss))
            self.writer.add_scalar('Train/Loss', epoch_loss, epoch)

            self.logger.info('Train ce-loss: {}'.format(epoch_ce_loss))
            self.writer.add_scalar('Train/CE-Loss', epoch_ce_loss, epoch)

            self.logger.info('Train dice-loss: {}'.format(epoch_dice_loss))
            self.writer.add_scalar('Train/Dice-Loss', epoch_dice_loss, epoch)


            self.logger.info('Train PS-loss: {}'.format(epoch_ps_loss))
            self.writer.add_scalar('Train/PS-Loss', epoch_ps_loss, epoch)

            self.logger.info('Train consistency-loss: {}'.format(epoch_consistency_loss))
            self.writer.add_scalar('Train/Consistency-Loss', epoch_consistency_loss, epoch)



            self.logger.info('Train IoU-1: {}'.format(epoch_iou_1))
            self.writer.add_scalar('Train/IoU-1', epoch_iou_1, epoch)

            self.logger.info('Train IoU-2: {}'.format(epoch_iou_2))
            self.writer.add_scalar('Train/IoU-2', epoch_iou_2, epoch)

            self.logger.info('Train Dice-1: {}'.format(epoch_dice_1))
            self.writer.add_scalar('Train/Dice-1', epoch_dice_1, epoch)

            self.logger.info('Train Dice-2: {}'.format(epoch_dice_2))
            self.writer.add_scalar('Train/Dice-2', epoch_dice_2, epoch)

            self.writer.add_scalar('info/lr', lr_, epoch)
            self.writer.add_scalar('info/consis_weight', consistency_weight, epoch)
            torch.cuda.empty_cache()

            self.model.eval()
            # self.model_2.eval()
            for i, pack in enumerate(val_loader, start=1):
                with torch.no_grad():
                    images, gts = pack
                    # images = Variable(images)
                    # gts = Variable(gts)
                    images = images.to(device)
                    gts = gts.to(device)
                    
                    prediction_1, prediction_2 = self.model(images)
                    # Prediction_1_soft = torch.softmax(prediction_1, dim=1)

                        

                # dice_coe_1 = dice_coef(prediction_1, gts)
                val_loss_ce_1 = ce_loss(prediction_1 , gts.long())
                # val_loss_ce_2 = ce_loss(prediction_2 , gts.long())
                val_loss_dice_1 = 1 - mDice(prediction_1, gts)
                # val_loss_dice_2 = 1 - mDice(prediction_2, gts)
                val_loss_1 = (val_loss_ce_1 + val_loss_dice_1)/2
                # val_loss_2 = (val_loss_ce_2 + val_loss_dice_2)/2

                val_loss = val_loss_1 #+ val_loss_2

                running_val_loss += val_loss.item()
                running_val_ce_loss += (val_loss_ce_1).item()
                running_val_dice_loss += (val_loss_dice_1).item()

                running_val_iou_1 += mIoU(prediction_1, gts)
                running_val_accuracy_1 += pixel_accuracy(prediction_1, gts)
                running_val_dice_1 += mDice(prediction_1, gts)

                 
            epoch_loss_val = running_val_loss / len(val_loader)
            epoch_ce_loss_val = running_val_ce_loss / len(val_loader)
            epoch_dice_loss_val = running_val_dice_loss / len(val_loader)
            epoch_dice_val_1 = running_val_dice_1 / len(val_loader)
            epoch_iou_val_1 = running_val_iou_1 / len(val_loader)
            epoch_accuracy_val_1 = running_val_accuracy_1 / len(val_loader)

            scheduler_1.step(epoch_dice_val_1)
            
            self.logger.info('Val loss: {}'.format(epoch_loss_val))
            self.writer.add_scalar('Validation/loss', epoch_loss_val, epoch)

            self.logger.info('Val CE loss: {}'.format(epoch_ce_loss_val))
            self.writer.add_scalar('Validation/ce-loss', epoch_ce_loss_val, epoch)
            self.logger.info('Val Dice loss: {}'.format(epoch_dice_loss_val))
            self.writer.add_scalar('Validation/dice-loss', epoch_dice_loss_val, epoch)

            #model-1 perfromance
            self.logger.info('Validation dice : {}'.format(epoch_dice_val_1))
            self.writer.add_scalar('Validation/mDice', epoch_dice_val_1, epoch)

            self.logger.info('Validation IoU : {}'.format(epoch_iou_val_1))
            self.writer.add_scalar('Validation/mIoU', epoch_iou_val_1, epoch)

            self.logger.info('Validation Accuracy : {}'.format(epoch_accuracy_val_1))
            self.writer.add_scalar('Validation/Accuracy', epoch_accuracy_val_1, epoch)

            
            mdice_coeff_1 =  epoch_dice_val_1
            # mdice_coeff_2 =  epoch_dice_val_2
            # mval_loss_1 = epoch_val_loss

            if self.best_dice_coeff_1 < mdice_coeff_1:
                self.best_dice_coeff_1 = mdice_coeff_1
                self.save_best_model_1 = True

                # if not os.path.exists(self.image_save_path_1):
                #     os.makedirs(self.image_save_path_1)

                # copy_tree(self.image_save_path_1, self.save_path + '/best_model_predictions_1')
                self.patience = 0
            else:
                self.save_best_model_1 = False
                self.patience += 1


                        
            Checkpoints_Path = self.save_path + '/Checkpoints'

            if not os.path.exists(Checkpoints_Path):
                os.makedirs(Checkpoints_Path)

            if self.save_best_model_1:
                state_1 = {
                "epoch": epoch,
                "best_dice_1": self.best_dice_coeff_1,
                "state_dict": self.model.state_dict(),
                "optimizer": optimizer_1.state_dict(),
                }
                # state["best_loss"] = self.best_loss
                torch.save(state_1, Checkpoints_Path + '/UCC_10p.pth')
  
 
            
            
             
            self.logger.info(
                'current best dice coef: model: {}'.format(self.best_dice_coeff_1))

            self.logger.info('current patience :{}'.format(self.patience))
            print('Current consistency weight:', consistency_weight)
            print('Current lr:', lr_)
      
            print('================================================================================================')
            print('================================================================================================')




if __name__ == '__main__':
    train_network = Network()
    train_network.run()