#-*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.legacy.nn as lnn
import torch.nn.functional as F
from torch.autograd import Variable
import random, pdb


##########################################################
# Models for predicting actions from states or latents
##########################################################

class LatentResidualModel3Layer(nn.Module):
    def __init__(self, opt):
        super(LatentResidualModel3Layer, self).__init__()
        self.opt = opt

        # deterministic network
        self.g_network_encoder = nn.Sequential(
            # layer 1
            nn.Conv2d(opt.n_in, opt.nfeature, 7, 2, 3), 
            nn.BatchNorm2d(opt.nfeature),
            nn.ReLU(True),
            # layer 2
            nn.Conv2d(opt.nfeature, opt.nfeature, 5, 2, 2), 
            nn.BatchNorm2d(opt.nfeature),
            nn.ReLU(True),
            # layer 3
            nn.Conv2d(opt.nfeature, opt.nfeature, 5, 2, 2), 
            nn.BatchNorm2d(opt.nfeature),
            nn.ReLU(True)
        )


        k = 4
        if opt.task == 'breakout' or opt.task == 'seaquest':
            # need this for output to be the right size
            k = 3

        self.g_network_decoder = nn.Sequential(
            # layer 4
            nn.ConvTranspose2d(opt.nfeature, opt.nfeature, k, 2, 1), 
            nn.BatchNorm2d(opt.nfeature),
            nn.ReLU(True),
            # layer 5
            nn.ConvTranspose2d(opt.nfeature, opt.nfeature, 4, 2, 1), 
            nn.BatchNorm2d(opt.nfeature),
            nn.ReLU(True),
            # layer 6
            nn.ConvTranspose2d(opt.nfeature, opt.n_out, 4, 2, 1) 
        )

        self.phi_network_conv = nn.Sequential(
            nn.Conv2d(opt.n_out, opt.nfeature, 7, 2, 3), 
            nn.BatchNorm2d(opt.nfeature),
            nn.ReLU(True),
            nn.Conv2d(opt.nfeature, opt.nfeature, 5, 2, 2), 
            nn.BatchNorm2d(opt.nfeature),
            nn.ReLU(True),
            nn.Conv2d(opt.nfeature, opt.nfeature, 5, 2, 2), 
            nn.BatchNorm2d(opt.nfeature), 
            nn.ReLU(True),
            nn.Conv2d(opt.nfeature, opt.nfeature, 5, 2, 2),  
            nn.BatchNorm2d(opt.nfeature), 
            nn.ReLU(True)
        )
        self.phi_network_fc = nn.Sequential(
            nn.Linear(opt.nfeature*opt.phi_fc_size, 1000), 
            nn.BatchNorm1d(1000), 
            nn.ReLU(True), 
            nn.Linear(1000, 1000),  
            nn.BatchNorm1d(1000), 
            nn.ReLU(True), 
            nn.Linear(1000, opt.n_latent), 
            nn.Tanh()
        )
            
        self.f_network_encoder = nn.Sequential(
            # layer 1
            nn.Conv2d(opt.n_in, opt.nfeature, 7, 2, 3),  
            nn.BatchNorm2d(opt.nfeature),
            nn.ReLU(True),
            # layer 2 
            nn.Conv2d(opt.nfeature, opt.nfeature, 5, 2, 2),  
            nn.BatchNorm2d(opt.nfeature),
            nn.ReLU(True),
            # layer 3
            nn.Conv2d(opt.nfeature, opt.nfeature, 5, 2, 2), 
            nn.BatchNorm2d(opt.nfeature), 
            nn.ReLU(True)
        )

        self.encoder_latent = nn.Linear(opt.n_latent, opt.nfeature) 

        self.f_network_decoder = nn.Sequential(
            # layer 1
            nn.ConvTranspose2d(opt.nfeature, opt.nfeature, k, 2, 1),  
            nn.BatchNorm2d(opt.nfeature),
            nn.ReLU(True),
            # layer 2
            nn.ConvTranspose2d(opt.nfeature, opt.nfeature, 4, 2, 1),  
            nn.BatchNorm2d(opt.nfeature),
            nn.ReLU(True),
            # layer 3
            nn.ConvTranspose2d(opt.nfeature, opt.n_out, 4, 2, 1) 
        )
        self.intype("gpu")
        

    # ultimate forward function
    def forward(self, input, target):
        input = input.view(self.opt.batch_size, self.opt.ncond*self.opt.nc, 
                                               self.opt.height, self.opt.width)
        target = target.view(self.opt.batch_size, self.opt.npred*self.opt.nc, 
                                               self.opt.height, self.opt.width)
        g_pred = self.g_network_decoder(self.g_network_encoder(input))
        # don't pass gradients to g from phi
        g_pred_v = Variable(g_pred.data)
        r = target - g_pred_v
        z = self.phi_network_fc(self.phi_network_conv(r).view(self.opt.batch_size, -1))
        z = z.view(self.opt.batch_size, self.opt.n_latent)
        z_emb = self.encoder_latent(z).view(input.size(0), self.opt.nfeature)
        s = self.f_network_encoder(input)
        h = s + z_emb.view(self.opt.batch_size, self.opt.nfeature, 1, 1).expand(s.size())
        pred_f = self.f_network_decoder(h)
        return pred_f, g_pred, z



    # generate a prediction given an input and z
    def decode(self, input, z):
        input = input.view(self.opt.batch_size, self.opt.ncond*self.opt.nc, 
                                               self.opt.height, self.opt.width)
        z_emb = self.encoder_latent(z).view(input.size(0), self.opt.nfeature)
        s = self.f_network_encoder(input)
        h = s + z_emb.view(self.opt.batch_size, self.opt.nfeature, 1, 1).expand(s.size())
        pred = self.f_network_decoder(h)
        return pred




    def intype(self, typ):
        if typ == "gpu":
            self.f_network_encoder.cuda()
            self.f_network_decoder.cuda()
            self.encoder_latent.cuda()
            self.g_network_encoder.cuda()
            self.g_network_decoder.cuda()
            self.phi_network_conv.cuda()
            self.phi_network_fc.cuda()
        elif typ == "cpu":
            self.f_network_encoder.cpu()
            self.f_network_decoder.cpu()
            self.encoder_latent.cpu()
            self.g_network_encoder.cpu()
            self.g_network_decoder.cpu()
            self.phi_network_conv.cpu()
            self.phi_network_fc.cpu()
        else:
            raise ValueError





# this model takes the true actions as input, to make sure 
# the architecture can make use of them
class GroundTruthModel(nn.Module):
    def __init__(self, opt):
        super(GroundTruthModel, self).__init__()
        self.opt = opt

        self.encoder_state = nn.Sequential(
            # layer 1
            nn.Conv2d(opt.n_in, opt.nfeature, 7, 2, 3),  # 64, 36 
            nn.BatchNorm2d(opt.nfeature),
            nn.ReLU(True),
            # layer 2 
            nn.Conv2d(opt.nfeature, opt.nfeature, 5, 2, 2),  # 32, 18
            nn.BatchNorm2d(opt.nfeature),
            nn.ReLU(True),
            # layer 3
            nn.Conv2d(opt.nfeature, opt.nfeature, 5, 2, 2),  # 16, 9
            nn.BatchNorm2d(opt.nfeature), 
            nn.ReLU(True),
            # layer 4
            nn.Conv2d(opt.nfeature, opt.nfeature, 5, 2, 2),  # 16, 9
            nn.BatchNorm2d(opt.nfeature)
        )

        # latent var (true actions)
        # TODO: fix actions
        self.encoder_latent = nn.Linear(opt.n_actions, 15*15*opt.latent_nfeature)  # TODO hard-coded
        self.encoder_latent_merge = nn.Sequential(
            nn.Conv2d(opt.latent_nfeature, opt.nfeature, 1, 1, 0),
            nn.BatchNorm2d(opt.nfeature)
        )

        self.fusion_network = nn.Sequential(
            # layer 1
            nn.ConvTranspose2d(opt.nfeature, opt.nfeature, 4, 2, 1),  # 32, 18
            nn.BatchNorm2d(opt.nfeature),
            nn.ReLU(True),
            # layer 2
            nn.ConvTranspose2d(opt.nfeature, opt.nfeature, 4, 2, 1),  # 32, 18
            nn.BatchNorm2d(opt.nfeature),
            nn.ReLU(True),
            # layer 3
            nn.ConvTranspose2d(opt.nfeature, opt.nfeature, 4, 2, 1),  # 64, 36
            nn.BatchNorm2d(opt.nfeature),
            nn.ReLU(True),
            # layer 4
            nn.ConvTranspose2d(opt.nfeature, 3, 4, 2, 1)  # 128, 72
        )


        
        # if enable GPU
        self.cuda = opt.gpu == 1
        if self.cuda:
            self.intype("gpu")
        else:
            self.intype("cpu")
        

    # ultimate forward function
    def forward(self, input, target, actions):
        target = target.view(self.opt.batch_size, self.opt.npred, 3, self.opt.height, self.opt.width)
        
        input = input.view(self.opt.batch_size, self.opt.ncond*3, 
                                               self.opt.height, self.opt.width)
        z = self.encoder_latent(actions).view(input.size(0),
                                                         self.opt.latent_nfeature, 15, 15)
        z = self.encoder_latent_merge(z)
        if self.opt.multiplicative == 0:
            pred = self.fusion_network(self.encoder_state(input) + z)
        else:
            pred = self.fusion_network(self.encoder_state(input) * z)
        
        return pred, z

    def intype(self, typ):
        if typ == "gpu":
            self.encoder_state.cuda()
            self.fusion_network.cuda()
            self.encoder_latent.cuda()
            self.encoder_latent_merge.cuda()
        elif typ == "cpu":
            self.encoder_state.cpu()
            self.fusion_network.cpu()
            self.encoder_latent.cpu()
            self.encoder_latent_merge.cpu()
        else:
            raise ValueError



class BaselineModel3Layer(nn.Module):
    def __init__(self, opt):
        super(BaselineModel3Layer, self).__init__()
        self.opt = opt

        # deterministic network
        self.f_network_encoder = nn.Sequential(
            # layer 1
            nn.Conv2d(opt.n_in, opt.nfeature, 7, 2, 3), 
            nn.BatchNorm2d(opt.nfeature),
            nn.ReLU(True),
            # layer 2
            nn.Conv2d(opt.nfeature, opt.nfeature, 5, 2, 2),  
            nn.BatchNorm2d(opt.nfeature),
            nn.ReLU(True),
            # layer 3
            nn.Conv2d(opt.nfeature, opt.nfeature, 5, 2, 2),  
            nn.BatchNorm2d(opt.nfeature),
            nn.ReLU(True)
        )

        k = 4
        if opt.task == 'breakout' or opt.task == 'seaquest':
            # need this for output to be the right size
            k = 3

        self.f_network_decoder = nn.Sequential(
            # layer 4
            nn.ConvTranspose2d(opt.nfeature, opt.nfeature, k, 2, 1), 
            nn.BatchNorm2d(opt.nfeature),
            nn.ReLU(True),
            # layer 5
            nn.ConvTranspose2d(opt.nfeature, opt.nfeature, 4, 2, 1), 
            nn.BatchNorm2d(opt.nfeature),
            nn.ReLU(True),
            # layer 6
            nn.ConvTranspose2d(opt.nfeature, opt.n_out, 4, 2, 1) 
        )
        self.intype("gpu")
        

    # ultimate forward function
    def forward(self, input):
        input = input.view(self.opt.batch_size, self.opt.n_in, self.opt.height, self.opt.width)
        h = self.f_network_encoder(input)
        pred = self.f_network_decoder(h)
        return pred

    def intype(self, typ):
        if typ == "gpu":
            self.f_network_encoder.cuda()
            self.f_network_decoder.cuda()
        elif typ == "cpu":
            self.f_network_encoder.cpu()
            self.f_network_decoder.cpu()
        else:
            raise ValueError


