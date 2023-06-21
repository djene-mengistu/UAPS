import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # specify which GPU(s) to be used
from datetime import datetime
from distutils.dir_util import copy_tree
# from torch.optim.lr_scheduler import ReduceLROnPlateau
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
from UAPS_dataloaders import*
# from new_dataloaders import*
# from loss import loss_sup, loss_adversarial_1, loss_adversarial_2, make_Dis_label, gt_label, loss_diff
from utilities.metrics import*
from utilities.losses_1 import*
from utilities.losses_2 import*
from utilities.pytorch_losses import dice_loss
from utilities.ramps import sigmoid_rampup
from UAPS_model import model
from utilities.utilities import get_logger, create_dir 
import os
seed = 1337
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # specify which GPU(s) to be used

parser = argparse.ArgumentParser() 
parser.add_argument('--model', type=str,
                    default='unet_ccps', help='model_name')
parser.add_argument('--num_classes', type=int,  default=6,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=20000, help='maximum epoch number to train')
parser.add_argument('--base_lr', type=float,  default=0.002,
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
num_classes = args.num_classes


kl_distance = nn.KLDivLoss(reduction='none') #KL_loss for consistency training
log_sm = torch.nn.LogSoftmax(dim = 1) #For computing the KL distance
ce_loss = CrossEntropyLoss()
# dice_loss = 1 - mDice(pred_mask, mask)
base_lr = args.base_lr
max_iterations = args.max_iterations
# kl_distance = nn.KLDivLoss(reduction='none')
iter_per_epoch = 20

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

        log_dir = '.../model_weights/MTiles/'

        self.logger = get_logger(log_dir)
        print('RUNDIR: {}'.format(log_dir))

        self.save_path = log_dir
        self.save_tbx_log = self.save_path + '/tbx_log'
        self.writer = SummaryWriter(self.save_tbx_log)

    def run(self):
        # print('Generator Learning Rate: {} Critic Learning Rate:{}'.format(lr_gen,lr_dis))
        self.model.to(device)
        optimizer_1 = torch.optim.Adam(self.model.parameters(), lr=base_lr)
        scheduler_1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_1, mode="max", min_lr = 0.00000001, patience=50, verbose=True)
      
        
        self.logger.info(
            "train_loader {} unlabeled_loader {} val_loader {} ".format(len(train_loader),
                                                                       len(unlabeled_loader),
                                                                       len(val_loader)))
        print("Training process started!")
        print("===============================================================================================")

        # model1.train()
        iter_num = 0
       
        for epoch in range(1, epochs):

            running_train_ce_loss = 0.0
            running_train_dice_loss = 0.0
            running_train_loss = 0.0
            running_train_iou = 0.0
            running_train_dice = 0.0
            running_train_ps_loss = 0.0
            running_md_loss = 0.0
            running_aux1_loss = 0.0
            running_aux2_loss = 0.0
            running_aux3_loss = 0.0
            running_train_consistency_loss = 0.0
            # running_train_lbl_consistency_loss = 0.0
            running_uncertainity_loss = 0.0

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

            semi_dataloader = iter(zip(cycle(train_loader), cycle(unlabeled_loader))) #Over-sampling the labeled samples 
                    
            for iteration in range (1, iter_per_epoch): #(zip(train_loader, unlabeled_train_loader)):
                
                data = next(semi_dataloader)
                
                (inputs_S1, labels_S1), (inputs_U, labels_U) = data #data[0][0], data[0][1]


                inputs_S1, labels_S1 = Variable(inputs_S1), Variable(labels_S1)
                inputs_S1, labels_S1 = inputs_S1.to(device), labels_S1.to(device)

                inputs_U, labels_U = Variable(inputs_U), Variable(labels_U)
                inputs_U, labels_U = inputs_U.to(device), labels_U.to(device)

                self.model.train()
                # self.model_2.train()

                # Train Model 1
                #Labeled samples output
                outputs, outputs_aux1, outputs_aux2, outputs_aux3 = self.model(inputs_S1)
                outputs_soft = torch.softmax(outputs, dim=1)
                outputs_aux1_soft = torch.softmax(outputs_aux1, dim=1)
                outputs_aux2_soft = torch.softmax(outputs_aux2, dim=1)
                outputs_aux3_soft = torch.softmax(outputs_aux3, dim=1)
                
                #Unlabeled samples output
                un_outputs, un_outputs_aux1, un_outputs_aux2, un_outputs_aux3 = self.model(inputs_U)
                un_outputs_soft = torch.softmax(un_outputs, dim=1)
                un_outputs_aux1_soft = torch.softmax(un_outputs_aux1, dim=1)
                un_outputs_aux2_soft = torch.softmax(un_outputs_aux2, dim=1)
                un_outputs_aux3_soft = torch.softmax(un_outputs_aux3, dim=1)
                
                
                #CE_loss
                loss_ce = ce_loss(outputs, labels_S1.long())
                loss_ce_aux1 = ce_loss(outputs_aux1, labels_S1.long())
                loss_ce_aux2 = ce_loss(outputs_aux2, labels_S1.long())
                loss_ce_aux3 = ce_loss(outputs_aux3, labels_S1.long())
                
                #Dice_loss
                loss_dice = dice_loss(labels_S1.unsqueeze(1), outputs)
                loss_dice_aux1 = dice_loss(labels_S1.unsqueeze(1), outputs_aux1)
                loss_dice_aux2 = dice_loss(labels_S1.unsqueeze(1), outputs_aux2)
                loss_dice_aux3 = dice_loss(labels_S1.unsqueeze(1), outputs_aux3)
                
                # loss_main = 0.5*(loss_ce + loss_dice)
                loss_main = 0.5*(loss_ce + loss_dice)
                loss_aux1 = 0.5*(loss_ce_aux1 + loss_dice_aux1)
                loss_aux2 = 0.5*(loss_ce_aux2 + loss_dice_aux2)
                loss_aux3 = 0.5*(loss_ce_aux3 + loss_dice_aux3)
                
                #Total supervised losss
                
                total_loss_ce = (loss_ce + loss_ce_aux1 + loss_ce_aux2 + loss_ce_aux3)/4  #for plotting epoch loss
                total_loss_dice = (loss_dice + loss_dice_aux1 + loss_dice_aux2 + loss_dice_aux3)/4  #for plotting epoch loss
                supervised_loss = (loss_main + loss_aux1 + loss_aux2 + loss_aux3)/4  #for plotting epoch loss
                
                
                # #Random weight for combining pridictions dynamically
                
                preds = (un_outputs_soft + un_outputs_aux1_soft + un_outputs_aux2_soft + un_outputs_aux3_soft)/4
                # # preds = (lbl_weight[0][0]*un_outputs_soft + lbl_weight[0][1]*un_outputs_aux1_soft + lbl_weight[0][2]*un_outputs_aux2_soft + lbl_weight[0][3]*un_outputs_aux3_soft)

                variance_main = torch.sum(kl_distance(log_sm(un_outputs), preds), dim=1)
                exp_variance_main = torch.exp(-variance_main)

                variance_aux1 = torch.sum(kl_distance(log_sm(un_outputs_aux1), preds), dim=1)
                exp_variance_aux1 = torch.exp(-variance_aux1)

                variance_aux2 = torch.sum(kl_distance(log_sm(un_outputs_aux2), preds), dim=1)
                exp_variance_aux2 = torch.exp(-variance_aux2)

                variance_aux3 = torch.sum(kl_distance(log_sm(un_outputs_aux3), preds), dim=1)
                exp_variance_aux3 = torch.exp(-variance_aux3)

                #Uncertainty loss
                ave_var = (variance_main + variance_aux1 + variance_aux2 + variance_aux3)/4  #Average variance
                exp_variance_ave = torch.exp(-ave_var) #Exponential average variance
                l_uncert = torch.mean(ave_var) #The uncertainty minimization loss

                #Consistency on unlabeled samples                            
                dist_1 = torch.mean((un_outputs_soft - un_outputs_aux1_soft) **2)
                dist_2 = torch.mean((un_outputs_soft - un_outputs_aux2_soft) **2)
                dist_3 = torch.mean((un_outputs_soft - un_outputs_aux3_soft) **2)

                # #Consistency with uncertainty
                # con_dist_1 = torch.mean(dist_1*exp_variance_aux1) + torch.mean(variance_aux1)
                # con_dist_2 = torch.mean(dist_2*exp_variance_aux2) + torch.mean(variance_aux2)
                # con_dist_3 = torch.mean(dist_3*exp_variance_aux3) + torch.mean(variance_aux3)
                
                # decoder_consistency_loss = (con_dist_1 + con_dist_2 + con_dist_3)/3
                decoder_consistency_loss = (dist_1 + dist_2 + dist_3)/3
                #Cross-pseudo labels

                #Pixel-level maximum pseudo labels 
                # un_lbl_pseudo = torch.argmax(torch.max(torch.max(un_outputs_aux1_soft.detach(), un_outputs_aux2_soft.detach()), un_outputs_aux3_soft.detach()), dim=1, keepdim=False)
                # lbl_pseudo = torch.argmax(torch.max(torch.max(outputs_aux1_soft.detach(), outputs_aux2_soft.detach()), outputs_aux3_soft.detach()), dim=1, keepdim=False)
                
                #Average pseudo-labels
                # un_lbl_pseudo = torch.argmax((un_outputs_soft.detach() + un_outputs_aux1_soft.detach() + un_outputs_aux2_soft.detach() + un_outputs_aux3_soft.detach())/4, dim=1, keepdim=False)
                
                # un_lbl_pseudo = torch.argmax((un_outputs_aux1_soft.detach() + un_outputs_aux2_soft.detach() + un_outputs_aux3_soft.detach())/3, dim=1, keepdim=False)# un_lbl_pseudo = torch.argmax(torch.max(un_outputs_soft.detach(), torch.max(un_outputs_aux1_soft.detach(), torch.max(un_outputs_aux2_soft.detach(),un_outputs_aux3_soft.detach()))), dim=1, keepdim=False) 
                # un_lbl_pseudo = torch.argmax(preds.detach(), dim=1, keepdim=False)
                # lbl_pseudo = torch.argmax((outputs_soft.detach() + outputs_aux1_soft.detach() + outputs_aux2_soft.detach() + outputs_aux3_soft.detach())/4, dim=1, keepdim=False)
                
                #Using dynamically mixed pseudo-labels:
                # lbl_weight = np.random.dirichlet(np.ones(4),size=1)
                lbl_weight = np.random.dirichlet(np.ones(4),size=1)[0]
                un_lbl_pseudo = torch.argmax((lbl_weight[0]*un_outputs_soft.detach() \
                                              + lbl_weight[1]*un_outputs_aux1_soft.detach() \
                                              + lbl_weight[2]*un_outputs_aux2_soft.detach() \
                                              + lbl_weight[3]*un_outputs_aux3_soft.detach()), dim=1, keepdim=False)
                # print (lbl_weight)
                # print('The sum is:', np.sum(lbl_weight))
                ps_main_loss = torch.mean(0.5*(ce_loss(un_outputs, un_lbl_pseudo) \
                    + dice_loss(un_lbl_pseudo.unsqueeze(1), un_outputs))*exp_variance_main) #unlabeled samples psuedo-label                
                ps_1_loss = torch.mean(0.5*(ce_loss(un_outputs_aux1, un_lbl_pseudo) \
                    + dice_loss(un_lbl_pseudo.unsqueeze(1), un_outputs_aux1))*exp_variance_aux1) #unlabeled samples psuedo-label
                ps_2_loss = torch.mean(0.5*(ce_loss(un_outputs_aux2, un_lbl_pseudo) \
                    + dice_loss(un_lbl_pseudo.unsqueeze(1), un_outputs_aux2))*exp_variance_aux2) #unlabeled samples psuedo-label
                ps_3_loss = torch.mean(0.5*(ce_loss(un_outputs_aux3, un_lbl_pseudo) \
                    + dice_loss(un_lbl_pseudo.unsqueeze(1), un_outputs_aux3))*exp_variance_aux3) #unlabeled samples psuedo-label


                # Pseudo-supervison loss without the uncertainty rectification

                # ps_main_loss = 0.5*(ce_loss(un_outputs, un_lbl_pseudo) + dice_loss(un_lbl_pseudo.unsqueeze(1), un_outputs))
                # ps_1_loss = 0.5*(ce_loss(un_outputs_aux1, un_lbl_pseudo) + dice_loss(un_lbl_pseudo.unsqueeze(1), un_outputs_aux1))
                # ps_2_loss = 0.5*(ce_loss(un_outputs_aux2, un_lbl_pseudo) + dice_loss(un_lbl_pseudo.unsqueeze(1), un_outputs_aux2))
                # ps_3_loss = 0.5*(ce_loss(un_outputs_aux3, un_lbl_pseudo) + dice_loss(un_lbl_pseudo.unsqueeze(1), un_outputs_aux3))
                                
               
                ps_loss = (ps_main_loss + ps_1_loss + ps_2_loss + ps_3_loss)/4
              
                consistency_weight = get_current_consistency_weight(iter_num // 50) #Consistency weight multipliers
                # mse_weight = 1*consistency_weight
                
                # loss = supervised_loss + mse_weight*decoder_consistency_loss + consistency_weight*ps_loss #+ consistency_weight*lbl_ps_loss
                loss = supervised_loss + 0*decoder_consistency_loss + consistency_weight*ps_loss + l_uncert
                
                
                optimizer_1.zero_grad()
                
                loss.backward()

                # if (i + 1 ) % self.accumulation_steps == 0:
                #     optimizer.step()
                #     optimizer.zero_grad()
                optimizer_1.step()
                # optimizer_2.step()
                # optimizer.zero_grad()
                running_train_loss += loss.item()
                running_train_ce_loss += total_loss_ce.item()
                running_train_dice_loss += total_loss_dice.item()
                running_train_ps_loss += ps_loss.item()
                running_train_consistency_loss += decoder_consistency_loss.item()
                # running_train_lbl_consistency_loss += lbl_decoder_consistency_loss.item()
                running_uncertainity_loss += l_uncert.item()
                running_md_loss += loss_ce.item()
                running_aux1_loss += loss_ce_aux1.item()
                running_aux2_loss += loss_ce_aux2.item()
                running_aux3_loss += loss_ce_aux3.item()
                running_train_iou += mIoU(outputs, labels_S1)
                running_train_dice += mDice(outputs, labels_S1)

                
                # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer_1.param_groups:
                    lr_ = param_group['lr']

                
                iter_num = iter_num + 1
            
            epoch_loss = (running_train_loss) / (iter_per_epoch)
            epoch_ce_loss = (running_train_ce_loss) / (iter_per_epoch)
            epoch_dice_loss = (running_train_dice_loss) / (iter_per_epoch)
            epoch_ps_loss = (running_train_ps_loss) / (iter_per_epoch)
            epoch_consistency_loss = (running_train_consistency_loss) / (iter_per_epoch)
            epoch_iou = (running_train_iou) / (iter_per_epoch)
            epoch_dice = (running_train_dice) / (iter_per_epoch)
            epoch_md_loss = (running_md_loss) / (iter_per_epoch)
            epoch_aux1_loss = (running_aux1_loss) / (iter_per_epoch)
            epoch_aux2_loss = (running_aux2_loss) / (iter_per_epoch)
            epoch_aux3_loss = (running_aux3_loss) / (iter_per_epoch)
            epoch_uncertainity_loss = (running_uncertainity_loss) / (iter_per_epoch)
            self.logger.info('{} Epoch [{:03d}/{:03d}], total_loss : {:.4f}'.
                             format(datetime.now(), epoch, epochs, epoch_loss))

            self.logger.info('Train loss: {}'.format(epoch_loss))
            self.writer.add_scalar('Train/Loss', epoch_loss, epoch)

            self.logger.info('Train ce-loss: {}'.format(epoch_ce_loss))
            self.writer.add_scalar('Train/CE-Loss', epoch_ce_loss, epoch)

            self.logger.info('Train dice-loss: {}'.format(epoch_dice_loss))
            self.writer.add_scalar('Train/Dice-Loss', epoch_dice_loss, epoch)

            self.logger.info('Train md-loss: {}'.format(epoch_md_loss))
            self.writer.add_scalar('Train/mdloss', epoch_md_loss, epoch)
            self.logger.info('Train aux1-loss: {}'.format(epoch_aux1_loss))
            self.writer.add_scalar('Train/aux1', epoch_aux1_loss, epoch)
            self.logger.info('Train aux2-loss: {}'.format(epoch_aux2_loss))
            self.writer.add_scalar('Train/aux2', epoch_aux2_loss, epoch)
            self.logger.info('Train aux3-loss: {}'.format(epoch_aux3_loss))
            self.writer.add_scalar('Train/aux3', epoch_aux3_loss, epoch)

            self.logger.info('Train PS-loss: {}'.format(epoch_ps_loss))
            self.writer.add_scalar('Train/PS-Loss', epoch_ps_loss, epoch)
            self.logger.info('Train consistency-loss: {}'.format(epoch_consistency_loss))
            self.writer.add_scalar('Train/consistency-Loss', epoch_consistency_loss, epoch)

            self.logger.info('Train uncertainity: {}'.format(epoch_uncertainity_loss))
            self.writer.add_scalar('Train/Uncertainity', epoch_uncertainity_loss, epoch)

            self.logger.info('Train IoU: {}'.format(epoch_iou))
            self.writer.add_scalar('Train/IoU', epoch_iou, epoch)
            self.logger.info('Train Dice: {}'.format(epoch_dice))
            self.writer.add_scalar('Train/Dice', epoch_dice, epoch)

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
                    
                    prediction_1, _, _, _ = self.model(images)
                    # Prediction_1_soft = torch.softmax(prediction_1, dim=1)

                        

                # dice_coe_1 = dice_coef(prediction_1, gts)
                loss_ce_1 = ce_loss(prediction_1, gts.long())
                loss_dice_1 = 1 - mDice(prediction_1, gts)
                # loss_ce = loss_ce_1 + loss_ce_2
                # loss_dice = loss_dice_1 + loss_dice_2

                val_loss = 0.5 * (loss_dice_1 + loss_ce_1)

                running_val_loss += val_loss.item()
                running_val_ce_loss += loss_ce_1.item()
                running_val_dice_loss += loss_dice_1.item()

                running_val_iou_1 += mIoU(prediction_1, gts)
                running_val_accuracy_1 += pixel_accuracy(prediction_1, gts)
                running_val_dice_1 += mDice(prediction_1, gts)

                # running_val_iou_2 += mIoU(prediction_2, gts)
                # running_val_accuracy_2 += pixel_accuracy(prediction_2, gts)
                # running_val_dice_2 += mDice(prediction_2, gts)
                
                 
            epoch_loss_val = running_val_loss / len(val_loader)
            epoch_ce_loss_val = running_val_ce_loss / len(val_loader)
            epoch_dice_loss_val = running_val_dice_loss / len(val_loader)
            epoch_dice_val_1 = running_val_dice_1 / len(val_loader)
            epoch_iou_val_1 = running_val_iou_1 / len(val_loader)
            epoch_accuracy_val_1 = running_val_accuracy_1 / len(val_loader)

            # epoch_dice_val_2 = running_val_dice_2 / len(val_loader)
            # epoch_iou_val_2 = running_val_iou_2 / len(val_loader)
            # epoch_accuracy_val_2 = running_val_accuracy_2 / len(val_loader)
            # scheduler.step(epoch_dice_val_1)
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
                torch.save(state_1, Checkpoints_Path + '/UAPS_MTiles_40p.pth')
  
 
            
            
             
            self.logger.info(
                'current best dice coef: model: {}'.format(self.best_dice_coeff_1))

            self.logger.info('current patience :{}'.format(self.patience))
            print('Current consistency weight:', consistency_weight)
            print('Current lr:', lr_)
            # print('pseudo mix weight:', lbl_weight)            
            print('================================================================================================')
            print('================================================================================================')




if __name__ == '__main__':
    train_network = Network()
    train_network.run()