from __future__ import division
import argparse, pdb, os, numpy, imp
from datetime import datetime
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import models, utils


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('-task', type=str, default='poke', help='breakout | seaquest | flappy | poke | driving')
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-model', type=str, default='vae2')
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-nfeature', type=int, default=64, help='number of feature maps')
parser.add_argument('-n_latent', type=int, default=4, help='dimensionality of z')
parser.add_argument('-lrt', type=float, default=0.0005, help='learning rate')
parser.add_argument('-epoch_size', type=int, default=500)
parser.add_argument('-loss', type=str, default='l2', help='l1 | l2')
parser.add_argument('-lambda_kl', type=float, default=0.001, help='weight of KL term')
parser.add_argument('-gpu', type=int, default=0)
parser.add_argument('-warmstart', type=int, default=1)
parser.add_argument('-datapath', type=str, default='/misc/vlgscratch4/LecunGroup/datasets/een_data/', help='data folder')
parser.add_argument('-save_dir', type=str, default='/misc/vlgscratch4/LecunGroup/mbhenaff/een_vae_xy/', help='where to save the models')
opt = parser.parse_args()

torch.manual_seed(opt.seed)
torch.set_default_tensor_type('torch.FloatTensor')
torch.cuda.set_device(opt.gpu)

if opt.task == 'poke':
    opt.loss = 'l1'

# load data and get dataset-specific parameters
data_config = utils.read_config('config.json').get(opt.task)
data_config['batchsize'] = opt.batch_size
data_config['datapath'] = '{}/{}'.format(opt.datapath, data_config['datapath'])
opt.ncond = data_config['ncond']
opt.npred = data_config['npred']
opt.height = data_config['height']
opt.width = data_config['width']
opt.nc = data_config['nc']
opt.phi_fc_size = data_config['phi_fc_size']
ImageLoader=imp.load_source('ImageLoader', 'dataloaders/{}.py'.format(data_config.get('dataloader'))).ImageLoader
dataloader = ImageLoader(data_config)

# Set filename based on parameters
opt.save_dir = '{}/{}/'.format(opt.save_dir, opt.task)
opt.model_filename = '{}/model={}-loss={}-ncond={}-npred={}-nf={}-nz={}-lrt={}-warmstart={}'.format(
                    opt.save_dir, opt.model, opt.loss, opt.ncond, opt.npred, opt.nfeature, opt.n_latent, opt.lrt, opt.warmstart)
print("Saving to " + opt.model_filename)



############
### train ##
############
def train_epoch(nsteps):
    total_loss_f, total_loss_kl = 0, 0
    model.train()
    for iter in range(0, nsteps):
        optimizer.zero_grad()
        model.zero_grad()
        cond, target, action = dataloader.get_batch('train')
        vcond = Variable(cond)
        vtarget = Variable(target)
        # forward
        pred_f, mu, logvar = model(vcond, vtarget)
        loss_f = criterion_f(pred_f, vtarget)
        total_loss_f += loss_f.data[0]
        loss_f.backward(retain_graph=True)
        loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss_kl /= opt.batch_size
        total_loss_kl += loss_kl.data[0]
        # optimize
        optimizer.step()
    return total_loss_f / nsteps, total_loss_kl / nsteps


def test_epoch(nsteps):
    total_loss_f, total_loss_kl = 0, 0
    model.eval()
    for iter in range(0, nsteps):
        cond, target, action = dataloader.get_batch('valid')
        vcond = Variable(cond)
        vtarget = Variable(target)
        pred_f, mu, logvar = model(vcond, vtarget)
        loss_f = criterion_f(pred_f, vtarget)
        total_loss_f += loss_f.data[0]
        loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss_kl += loss_kl.data[0]

    return total_loss_f / nsteps, total_loss_kl / nsteps

def train(n_epochs):
    # prepare for saving 
    os.system("mkdir -p " + opt.save_dir)
    # training
    best_valid_loss_f = 1e6
    train_loss_f, train_loss_kl = [], []
    valid_loss_f, valid_loss_kl = [], []
    for i in range(0, n_epochs):
        train_loss_epoch_f, train_loss_epoch_g = train_epoch(opt.epoch_size)
        train_loss_f.append(train_loss_epoch_f)
        train_loss_kl.append(train_loss_epoch_g)
        valid_loss_epoch_f, valid_loss_epoch_g = test_epoch(int(opt.epoch_size / 5))
        valid_loss_f.append(valid_loss_epoch_f)
        valid_loss_kl.append(valid_loss_epoch_g)

        if valid_loss_f[-1] < best_valid_loss_f:
            best_valid_loss_f = valid_loss_f[-1]
            # save the whole model
            model.intype("cpu")
            torch.save({ 'i': i, 'model': model, 'train_loss_f': train_loss_f, 'train_loss_kl': train_loss_kl, 'valid_loss_f': valid_loss_f, 'valid_loss_kl': valid_loss_kl},
                       opt.model_filename + '.model')
            torch.save(optimizer, opt.model_filename + '.optim')
            model.intype("gpu")

        log_string = ('iter: {:d}, train_loss_f: {:0.6f}, train_loss_kl: {:0.6f}, valid_loss_f: {:0.6f}, valid_loss_kl: {:0.6f}, best_valid_loss_f: {:0.6f}, lr: {:0.5f}').format(
                      (i+1)*opt.epoch_size, train_loss_f[-1], train_loss_kl[-1], valid_loss_f[-1], valid_loss_kl[-1], best_valid_loss_f, opt.lrt)
        print(log_string)
        utils.log(opt.model_filename + '.log', log_string)


if __name__ == '__main__':
    numpy.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    # build the model
    opt.n_in = opt.ncond * opt.nc
    opt.n_out = opt.npred * opt.nc
    if opt.model == 'vae':
        model = models.VAE(opt)
    elif opt.model == 'vae2':
        model = models.VAE2(opt)

    if opt.warmstart == 1:
        # load the baseline model and copy its weights
        mdir = '/misc/vlgscratch4/LecunGroup/mbhenaff/een_release_results/{}/'.format(opt.task)
        mfile = 'model=baseline-3layer-loss={}-ncond={}-npred={}-nf={}-lrt=0.0005.model'.format(opt.loss, opt.ncond, opt.npred, opt.nfeature)
        print('initializing with baseline model: {}'.format(mdir + mfile))
        baseline_model = torch.load(mdir + mfile).get('model')
        model.f_network_encoder.load_state_dict(baseline_model.f_network_encoder.state_dict())
        model.f_network_decoder.load_state_dict(baseline_model.f_network_decoder.state_dict())


    optimizer = optim.Adam(model.parameters(), opt.lrt)

    if opt.loss == 'l1':
        criterion_f = nn.L1Loss().cuda()
    elif opt.loss == 'l2':
        criterion_f = nn.MSELoss().cuda()
    print('training...')
    utils.log(opt.model_filename + '.log', '[training]')
    train(500)

