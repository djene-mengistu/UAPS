import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # specify which GPU(s) to be used
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
from utilities.dataloaders import*
# from new_dataloaders import*
# from loss import loss_sup, loss_adversarial_1, loss_adversarial_2, make_Dis_label, gt_label, loss_diff
from utilities.metrics import*
from utilities.losses_1 import*
from utilities.losses_2 import*
from utilities.pytorch_losses import dice_loss
from utilities.ramps import sigmoid_rampup
from UA_MT_model import model, ema_model #, model2 #, Critic_model
from utilities.utilities import get_logger, create_dir 
seed = 1337
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # specify which GPU(s) to be used

parser = argparse.ArgumentParser() 
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=36010, help='maximum epoch number to train')
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


epochs = 600
# batchsize = 16 
# CE = torch.nn.BCELoss()
# criterion_1 = torch.nn.BCELoss()
num_classes = args.num_classes

ce_loss = CrossEntropyLoss()
# dice_loss = 1 - mDice(pred_mask, mask)
base_lr = args.base_lr
max_iterations = args.max_iterations

iter_per_epoch = 30

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


class Network(object):
    def __init__(self):
        self.patience = 0
        self.best_dice_coeff_1 = False
        self.best_dice_coeff_2 = False
        self.model = model
        self.ema_model = ema_model 
        self._init_logger()

    # def _init_configure(self):
    #     with open('configs/config.yml') as fp:
    #         self.cfg = yaml.safe_load(fp)

    def _init_logger(self):

        log_dir = '.../model_weights/NEU_seg/'

        self.logger = get_logger(log_dir)
        print('RUNDIR: {}'.format(log_dir))

        self.save_path = log_dir
        # self.image_save_path_1 = log_dir + "/saved_images_1"
        # self.image_save_path_2 = log_dir + "/saved_images_2"

        # create_dir(self.image_save_path_1)
        # create_dir(self.image_save_path_2)

        self.save_tbx_log = self.save_path + '/tbx_log'
        self.writer = SummaryWriter(self.save_tbx_log)

    def run(self): 

        self.model.to(device)
        self.ema_model.to(device)
        optimizer_1 = torch.optim.Adam(self.model.parameters(), lr=base_lr)
        # optimizer = torch.optim.Adam(params, lr=lr_gen)
        scheduler_1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_1, mode="max", min_lr = 0.0000001, patience=40, verbose=True)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=20, verbose=True)
      
        
        self.logger.info(
            "train_loader {} unlabeled_loader {} val_loader {}".format(len(train_loader),
                                                                       len(unlabeled_loader),
                                                                       len(val_loader)))
        print("Training process started!")
        print("===============================================================================================")

        # model1.train()
        iter_num = 0
       
        for epoch in range(1, epochs):

            running_ce_loss = 0.0
            running_dice_loss = 0.0
            running_train_loss = 0.0
            running_train_consistency_loss = 0.0
            
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
                
                (inputs_S1, labels_S1), (inputs_U, labels_U) = data #data[0][0], data[0][1]


                inputs_S1, labels_S1 = Variable(inputs_S1), Variable(labels_S1)
                inputs_S1, labels_S1 = inputs_S1.to(device), labels_S1.to(device)

                inputs_U, labels_U = Variable(inputs_U), Variable(labels_U)
                inputs_U, labels_U = inputs_U.to(device), labels_U.to(device)

                self.model.train()
                # self.model_2.train()

                # Train Model 1
                outputs_1 = self.model(inputs_S1)
                outputs_soft_1 = torch.softmax(outputs_1, dim=1)
                
                un_outputs_1 = self.model(inputs_U)
                un_outputs_soft_1 = torch.softmax(un_outputs_1, dim=1)

                noise = torch.clamp(torch.randn_like(inputs_U) * 0.1, -0.2, 0.2)
                ema_inputs = inputs_U + noise

                
                with torch.no_grad():
                    ema_output = self.ema_model(ema_inputs)
                    ema_output_soft = torch.softmax(ema_output, dim=1)

                T = 8
                _, _, w, h = inputs_U.shape
                u_batch_r = inputs_U.repeat(2, 1, 1, 1)
                stride = u_batch_r.shape[0] // 2
                preds = torch.zeros([stride * T, num_classes, w, h]).cuda()
                for i in range(T//2):
                    ema_inputs = u_batch_r + torch.clamp(torch.randn_like(u_batch_r) * 0.1, -0.2, 0.2)
                    with torch.no_grad():
                        preds[2 * stride * i:2 * stride * (i + 1)] = self.ema_model(ema_inputs)
                preds = F.softmax(preds, dim=1)
                preds = preds.reshape(T, stride, num_classes, w, h)
                preds = torch.mean(preds, dim=0)
                uncertainty = -1.0 * torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True)
                
                loss_ce_1 = ce_loss(outputs_1, labels_S1.long())
                loss_dice_1 = dice_loss(labels_S1.unsqueeze(1), outputs_1)
                
                loss_sup = 0.5 * (loss_dice_1 + loss_ce_1)

                consistency_weight = get_current_consistency_weight(iter_num//80)
                
               
                consistency_dist = softmax_mse_loss(un_outputs_1, ema_output)  # (batch, 2, 112,112,80)
                # threshold = (0.75 + 0.25*sigmoid_rampup(iter_num, max_iterations))*np.log(2)
                threshold = (0.75 + 2.5*consistency_weight)*np.log(2)
                mask = (uncertainty < threshold).float()
                consistency_loss = torch.sum(mask*consistency_dist)/(2*torch.sum(mask) + 1e-16)
                loss = loss_sup + consistency_weight * consistency_loss
             
                # seg_loss = seg_loss / self.accumulation_steps
                optimizer_1.zero_grad()
                
                loss.backward()

                # if (i + 1 ) % self.accumulation_steps == 0:
                #     optimizer.step()
                #     optimizer.zero_grad()
                optimizer_1.step()
                # optimizer_2.step()
                # optimizer.zero_grad()
                running_train_loss += loss.item()
                running_ce_loss += loss_ce_1.item()
                running_dice_loss += loss_dice_1.item()
                running_train_consistency_loss += consistency_loss.item()

                update_ema_variables(self.model, self.ema_model, args.ema_decay, iter_num)

                # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer_1.param_groups:
                    lr_ = param_group['lr']

                
                iter_num = iter_num + 1
            
            epoch_loss = (running_train_loss) / (len(train_loader))
            epoch_ce_loss = (running_ce_loss) / (len(train_loader))
            epoch_dice_loss = (running_dice_loss) / (len(train_loader))
            epoch_consistency_loss = (running_train_consistency_loss)/ (len(train_loader))
            # epoch_loss = running_loss / (len(train_loader))
            self.logger.info('{} Epoch [{:03d}/{:03d}], total_loss : {:.4f}'.
                             format(datetime.now(), epoch, epochs, epoch_loss))

            self.logger.info('Train loss: {}'.format(epoch_loss))
            self.writer.add_scalar('Train/Loss', epoch_loss, epoch)

            self.logger.info('Train ce-loss: {}'.format(epoch_ce_loss))
            self.writer.add_scalar('Train/CE-Loss', epoch_ce_loss, epoch)

            self.logger.info('Train dice-loss: {}'.format(epoch_dice_loss))
            self.writer.add_scalar('Train/Dice-Loss', epoch_dice_loss, epoch)

            self.logger.info('Train consistency-loss: {}'.format(epoch_consistency_loss))
            self.writer.add_scalar('Train/Con-Loss', epoch_consistency_loss, epoch)

            self.writer.add_scalar('info/lr', lr_, epoch)
            self.writer.add_scalar('info/consis_weight', consistency_weight, epoch)
            self.writer.add_scalar('info/threshold', threshold, epoch)
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
                    
                    prediction_1 = self.model(images)
                    # Prediction_1_soft = torch.softmax(prediction_1, dim=1)

                        

                # dice_coe_1 = dice_coef(prediction_1, gts)
                loss_ce_1 = ce_loss(prediction_1, gts.long())
                loss_dice_1 = 1 - mDice(prediction_1, gts)
                # loss_ce = loss_ce_1 + loss_ce_2
                # loss_dice = loss_dice_1 + loss_dice_2

                val_loss = 0.5 * (loss_dice_1 + loss_ce_1)

                running_val_loss += val_loss
                running_val_iou_1 += mIoU(prediction_1, gts)
                running_val_accuracy_1 += pixel_accuracy(prediction_1, gts)
                running_val_dice_1 += mDice(prediction_1, gts)

                # running_val_iou_2 += mIoU(prediction_2, gts)
                # running_val_accuracy_2 += pixel_accuracy(prediction_2, gts)
                # running_val_dice_2 += mDice(prediction_2, gts)
                
                 
            epoch_loss_val = running_val_loss / len(val_loader)
            epoch_dice_val_1 = running_val_dice_1 / len(val_loader)
            epoch_iou_val_1 = running_val_iou_1 / len(val_loader)
            epoch_accuracy_val_1 = running_val_accuracy_1 / len(val_loader)

            # epoch_dice_val_2 = running_val_dice_2 / len(val_loader)
            # epoch_iou_val_2 = running_val_iou_2 / len(val_loader)
            # epoch_accuracy_val_2 = running_val_accuracy_2 / len(val_loader)
            scheduler_1.step(epoch_dice_val_1)
            
            self.logger.info('Val loss: {}'.format(epoch_loss_val))
            self.writer.add_scalar('Validation/loss', epoch_loss_val, epoch)

            #model-1 perfromance
            self.logger.info('Validation dice_1 : {}'.format(epoch_dice_val_1))
            self.writer.add_scalar('Validation/DSC-1', epoch_dice_val_1, epoch)

            self.logger.info('Validation IoU_1 : {}'.format(epoch_iou_val_1))
            self.writer.add_scalar('Validation/IoU-1', epoch_iou_val_1, epoch)

            self.logger.info('Validation Accuracy_1 : {}'.format(epoch_accuracy_val_1))
            self.writer.add_scalar('Validation/Accuracy-1', epoch_accuracy_val_1, epoch)

            
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
                torch.save(state_1, Checkpoints_Path + '/UA_MT__10p.pth')
  
 
            
            
             
            self.logger.info(
                'current best dice coef: model: {}'.format(self.best_dice_coeff_1))

            self.logger.info('current patience :{}'.format(self.patience))
            print('Current consistency weight:', consistency_weight)
            print('Current threshold:', threshold)
            print('================================================================================================')
            print('================================================================================================')




if __name__ == '__main__':
    train_network = Network()
    train_network.run()