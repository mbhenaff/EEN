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
parser.add_argument('-model', type=str, default='baseline-3layer')
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-nfeature', type=int, default=64, help='number of feature maps in convnet')
parser.add_argument('-lrt', type=float, default=0.0005, help='learning rate')
parser.add_argument('-epoch_size', type=int, default=500)
parser.add_argument('-loss', type=str, default='l2', help='l1 | l2')
parser.add_argument('-gpu', type=int, default=1)
parser.add_argument('-datapath', type=str, default='/misc/vlgscratch4/LecunGroup/datasets/een_data/')
parser.add_argument('-save_dir', type=str, default='./results/', help='where to save the models')
opt = parser.parse_args()

torch.manual_seed(opt.seed)
torch.set_default_tensor_type('torch.FloatTensor')
torch.cuda.set_device(opt.gpu)


# load data and get dataset-specific parameters
data_config = utils.read_config('config.json').get(opt.task)
data_config['batchsize'] = opt.batch_size
data_config['datapath'] = '{}/{}'.format(opt.datapath, data_config['datapath'])
opt.ncond = data_config['ncond']
opt.npred = data_config['npred']
opt.height = data_config['height']
opt.width = data_config['width']
opt.nc = data_config['nc']
ImageLoader=imp.load_source('ImageLoader', 'dataloaders/{}.py'.format(data_config.get('dataloader'))).ImageLoader
dataloader = ImageLoader(data_config)


# Set filename based on parameters
opt.save_dir = '{}/{}/'.format(opt.save_dir, opt.task)
opt.model_filename = '{}/model={}-loss={}-ncond={}-npred={}-nf={}-lrt={}'.format(
                    opt.save_dir, opt.model, opt.loss, opt.ncond, opt.npred, opt.nfeature, opt.lrt)
print("Saving to " + opt.model_filename)




############
### train ##
############
def train_epoch(nsteps):
    total_loss = 0
    model.train()
    for iter in range(0, nsteps):
        optimizer.zero_grad()
        model.zero_grad()
        cond, target, _ = dataloader.get_batch('train')
        vcond = Variable(cond)
        vtarget = Variable(target)
        # forward
        pred = model(vcond)
        loss = criterion(pred, vtarget)
        total_loss += loss.data[0]
        loss.backward()
        optimizer.step()
    return total_loss / nsteps


def test_epoch(nsteps):
    total_loss = 0
    model.eval()
    for iter in range(0, nsteps):
        cond, target, action = dataloader.get_batch('valid')
        vcond = Variable(cond)
        vtarget = Variable(target)
        pred = model(vcond)
        loss = criterion(pred, vtarget)
        total_loss += loss.data[0]
    return total_loss / nsteps

def train(n_epochs):
    # prepare for saving 
    os.system("mkdir -p " + opt.save_dir)
    # training
    best_valid_loss = 1e6
    train_loss, valid_loss = [], []
    for i in range(0, n_epochs):        
        train_loss.append(train_epoch(opt.epoch_size))
        valid_loss.append(test_epoch(opt.epoch_size))

        if valid_loss[-1] < best_valid_loss:
            best_valid_loss = valid_loss[-1]
            # save 
            model.intype("cpu")
            torch.save({ 'epoch': i, 'model': model, 'train_loss': train_loss, 'valid_loss': valid_loss},
                       opt.model_filename + '.model')
            torch.save(optimizer, opt.model_filename + '.optim')
            model.intype("gpu")

        log_string = ('iter: {:d}, train_loss: {:0.6f}, valid_loss: {:0.6f}, best_valid_loss: {:0.6f}, lr: {:0.5f}').format(
                      (i+1)*opt.epoch_size, train_loss[-1], valid_loss[-1], best_valid_loss, opt.lrt)
        print(log_string)
        utils.log(opt.model_filename + '.log', log_string)


if __name__ == '__main__':
    numpy.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    # build the model
    opt.n_in = opt.ncond * opt.nc
    opt.n_out = opt.npred * opt.nc
    model = models.BaselineModel3Layer(opt).cuda()
    optimizer = optim.Adam(model.parameters(), opt.lrt)
    if opt.loss == 'l1':
        criterion = nn.L1Loss().cuda()
    elif opt.loss == 'l2':
        criterion = nn.MSELoss().cuda()
    print('training...')
    utils.log(opt.model_filename + '.log', '[training]')
    train(500)

