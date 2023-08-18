from config import *
from model import *
from dataset import DataSet 
from logger import Log

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from math import pi, cos
from tqdm import tqdm

from module.gcn.st_gcn import Model

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(0)

# %%
class Processor:

    @ex.capture
    def load_data(self, train_list, train_label, test_list, test_label, batch_size, language_path):
        self.dataset = dict()
        self.data_loader = dict()
        self.best_epoch = -1
        self.best_acc = -1
        self.dim_loss = -1
        self.test_acc = -1
        
        self.full_language = np.load(language_path)
        self.full_language = torch.Tensor(self.full_language)
        self.full_language = F.normalize(self.full_language, dim=-1)
        self.full_language = self.full_language.cuda()
        self.dataset['train'] = DataSet(train_list, train_label)
        self.dataset['test'] = DataSet(test_list, test_label)

        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=self.dataset['train'],
            batch_size=batch_size,
            num_workers=16,
            shuffle=True)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=self.dataset['test'],
            batch_size=64,
            num_workers=16,
            shuffle=False)

    def load_weights(self, model=None, weight_path=None):
        pretrained_dict = torch.load(weight_path)
        model.load_state_dict(pretrained_dict)

    def adjust_learning_rate(self,optimizer,current_epoch, max_epoch,lr_min=0,lr_max=0.1,warmup_epoch=15):

        if current_epoch < warmup_epoch:
            lr = lr_max * current_epoch / warmup_epoch
        else:
            lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    def layernorm(self, feature):

        num = feature.shape[0]
        mean = torch.mean(feature, dim=1).reshape(num, -1)
        var = torch.var(feature, dim=1).reshape(num, -1)
        out = (feature-mean) / torch.sqrt(var)

        return out

    @ex.capture
    def load_model(self,in_channels,hidden_channels,hidden_dim,
                    dropout,graph_args,edge_importance_weighting, visual_size, language_size, weight_path):
        self.encoder = Model(in_channels=in_channels, hidden_channels=hidden_channels,
                            hidden_dim=hidden_dim,dropout=dropout, 
                            graph_args=graph_args,
                            edge_importance_weighting=edge_importance_weighting,
                            )
        self.encoder = self.encoder.cuda()
        self.model = MI(visual_size, language_size).cuda()
        self.load_weights(self.encoder, weight_path)

    @ex.capture
    def load_optim(self, lr, epoch_num, weight_decay):
        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.model.parameters()}],
             lr=lr,
             weight_decay=weight_decay,
             )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 100)

    @ex.capture
    def optimize(self, epoch_num):
        print("main track")
        for epoch in range(epoch_num):
            self.train_epoch(epoch)
            with torch.no_grad():
                self.test_epoch(epoch=epoch)
            print("epoch [{}] dim loss: {}".format(epoch,self.dim_loss))
            if epoch > 15:
                print("epoch [{}] test acc: {}".format(epoch,self.test_acc))
                print("epoch [{}] gets the best acc: {}".format(self.best_epoch,self.best_acc))
            else:
                print("epoch [{}] : warm up epoch.".format(epoch))

    @ex.capture
    def train_epoch(self, epoch, lr, margin):
        self.encoder.eval()
        self.model.train()
        self.adjust_learning_rate(self.optimizer, current_epoch=epoch, max_epoch=100, lr_max=lr)
        running_loss = []
        loader = self.data_loader['train']
        for data, label in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            seen_language = self.full_language[label]
            
            # Global
            input0 = data.clone()
            feat0 = self.encoder(input0).detach()
            dim0 = self.model(feat0, seen_language)

            # Temporal
            input1 = motion_att_temp_mask2(data, 15)
            feat1 = self.encoder(input1).detach()
            dim1 = self.model.temp_constrain(feat0, feat1, seen_language)

            # Margin
            beta = margin

            # Loss
            loss1 = -dim0
            loss2 = torch.clamp(beta-(dim0-dim1), min=0)
            
            #loss = loss1
            loss = loss1 + 0.5*loss2

            running_loss.append(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        running_loss = torch.tensor(running_loss)
        self.dim_loss = running_loss.mean().item()

    @ex.capture
    def test_epoch(self, unseen_label, epoch):
        self.encoder.eval()
        self.model.eval()

        loader = self.data_loader['test']
        y_true = []
        y_pred = []
        acc_list = []
        for data, label in tqdm(loader):

            # y_t = label.numpy().tolist()
            # y_true += y_t

            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            unseen_language = self.full_language[unseen_label]
            # inference
            feature = self.encoder(data)
            acc_batch, pred = self.model.get_acc(feature, unseen_language, label)

            # y_p = pred.cpu().numpy().tolist()
            # y_pred += y_p


            acc_list.append(acc_batch)
        acc_list = torch.tensor(acc_list)
        acc = acc_list.mean()
        if epoch>15 and acc > self.best_acc:
            self.best_acc = acc
            self.best_epoch = epoch
            # y_true = np.array(y_true)
            # y_pred = np.array(y_pred)
            # np.save("y_true_3.npy",y_true)
            # np.save("y_pred_3.npy",y_pred)
            # print("save ok!")
        self.test_acc = acc


    def initialize(self):
        self.load_data()
        self.load_model()
        self.load_optim()
        self.log = Log()

    @ex.capture
    def save_model(self, save_path):
        torch.save(self.model, save_path)

    def start(self):
        self.initialize()
        self.optimize()
        self.save_model()

class SotaProcessor:

    @ex.capture
    def load_data(self, sota_train_list, sota_train_label, 
        sota_test_list, sota_test_label, batch_size, language_path):
        self.dataset = dict()
        self.data_loader = dict()
        self.best_epoch = -1
        self.best_acc = -1
        self.dim_loss = -1
        self.test_acc = -1
         
        self.full_language = np.load(language_path)
        self.full_language = torch.Tensor(self.full_language)
        self.full_language = F.normalize(self.full_language,dim=-1)
        self.full_language = self.full_language.cuda()

        self.dataset['train'] = DataSet(sota_train_list, sota_train_label)
        self.dataset['test'] = DataSet(sota_test_list, sota_test_label)

        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=self.dataset['train'],
            batch_size=batch_size,
            num_workers=16,
            shuffle=True)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=self.dataset['test'],
            batch_size=64,
            num_workers=16,
            shuffle=False)

    def adjust_learning_rate(self,optimizer,current_epoch, max_epoch,lr_min=0,lr_max=0.1,warmup_epoch=15):

            if current_epoch < warmup_epoch:
                lr = lr_max * current_epoch / warmup_epoch
            else:
                lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    @ex.capture
    def load_model(self,in_channels,hidden_channels,hidden_dim,
                    dropout,graph_args,edge_importance_weighting, visual_size, language_size, weight_path):
        self.model = MI(visual_size, language_size).cuda()

    @ex.capture
    def load_optim(self, lr, epoch_num, weight_decay):
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()}],
             lr=lr,
             weight_decay=weight_decay,
             )

    @ex.capture
    def optimize(self, epoch_num):
        print("sota track")
        for epoch in range(epoch_num):
            self.train_epoch(epoch)
            with torch.no_grad():
                self.test_epoch(epoch=epoch)
            print("epoch [{}] dim loss: {}".format(epoch,self.dim_loss))
            print("epoch [{}] test acc: {}".format(epoch,self.test_acc))
            print("epoch [{}] gets the best acc: {}".format(self.best_epoch,self.best_acc))

    @ex.capture
    def train_epoch(self, epoch, lr):
        self.model.train()
        self.adjust_learning_rate(self.optimizer, current_epoch=epoch, max_epoch=100, lr_max=lr)
        running_loss = []
        loader = self.data_loader['train']
        for data, label in tqdm(loader):

            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            seen_language = self.full_language[label]
            
            # Global
            feat0 = data.clone()
            dim = self.model(feat0, seen_language)

            # Loss
            loss = -dim
            
            running_loss.append(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        running_loss = torch.tensor(running_loss)
        self.dim_loss = running_loss.mean().item()

    @ex.capture
    def test_epoch(self, sota_unseen, epoch):
        self.model.eval()

        total = 0
        correct = 0
        loader = self.data_loader['test']
        acc_list = []
        for data, label in tqdm(loader):
            feature = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            unseen_language = self.full_language[sota_unseen]
            # inference
            acc_batch = self.model.get_acc(feature, unseen_language, label, sota_unseen)
            acc_list.append(acc_batch)
        acc_list = torch.tensor(acc_list)
        acc = acc_list.mean()
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_epoch = epoch
        self.test_acc = acc

    def initialize(self):
        self.load_data()
        self.load_model()
        self.load_optim()
        self.log = Log()

    def start(self):
        self.initialize()
        self.optimize()

# %%
@ex.automain
def main(track):
    if "sota" in track:
        p = SotaProcessor()
    elif "main" in track:
        p = Processor()
    p.start()
