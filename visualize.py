import argparse, pdb, os, numpy, glob, imp, imageio
from datetime import datetime
import matplotlib as mpi
mpi.use('Agg')
import matplotlib.pyplot as plt
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import models, utils
from sklearn.decomposition import PCA


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('-task', type=str, default='poke')
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-gpu', type=int, default=1)
parser.add_argument('-datapath', type=str, default='./data/')
parser.add_argument('-save_dir', type=str,
                     default='./results/')
opt = parser.parse_args()
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

opt.save_dir = '{}/{}/'.format(opt.save_dir, opt.task)


def load_model(mfile):
    model = torch.load(mfile).get('model')
    model = model.cuda()
    model.eval()
    return model


# sample or load a presaved batch so that all models are evaluated on the same samples
fname = 'data_for_viz/{}/data-ncond={}-npred={}.th'.format(opt.task, opt.ncond, opt.npred)
if os.path.isfile(fname):
    print('loading previous batch')
    loaded = torch.load(fname)
    cond = loaded.get('cond')
    target = loaded.get('target')
    action = loaded.get('action')
else:
    cond, target, action = dataloader.get_batch('test')
    if not os.path.isdir(os.path.dirname(fname)):
            os.system("mkdir -p " + os.path.dirname(fname))
    torch.save({'cond': cond, 'target': target, 'action': action}, fname)

vcond = Variable(cond.cuda())
vtarget = Variable(target.cuda())


#################################
# Baseline Model Predictions
#################################

for mfile in glob.glob(opt.save_dir + '/*model=baseline*.model'):
    print('loading {}'.format(mfile))
    model = load_model(mfile)

    # make folder to save visualizations
    save_dir = mfile + '.viz'
    if not os.path.isdir(save_dir):
        os.system('mkdir -p ' + save_dir)

    # get predictions for this batch
    pred = model(vcond)
    pred = pred.data.view(opt.batch_size, opt.npred, opt.nc, opt.height, opt.width)
    cond = cond.view(opt.batch_size, opt.ncond, opt.nc, opt.height, opt.width)
    target = target.view(opt.batch_size, opt.npred, opt.nc, opt.height, opt.width)
    # save each prediction along with the ground truth
    for b in range(opt.batch_size):
        img = dataloader.plot_seq(cond[b].cpu().unsqueeze(0), pred[b].cpu().unsqueeze(0))
        fname = '{}/ep{}_baseline.png'.format(save_dir, b)
        torchvision.utils.save_image(img, fname)
        img = dataloader.plot_seq(cond[b].cpu().unsqueeze(0), target[b].cpu().unsqueeze(0))
        fname = '{}/ep{}_truth.png'.format(save_dir, b)
        torchvision.utils.save_image(img, fname)



###############################
# Latent Model Predictions
###############################

for mfile in glob.glob(opt.save_dir + '/*latent*.model'.format(opt.npred)):
    print('loading {}'.format(mfile))
    model = load_model(mfile)

    # make folder to save visualizations
    save_dir = mfile + '.viz'
    if not os.path.isdir(save_dir):
        os.system('mkdir -p ' + save_dir)

    # extract some z vectors from the training set
    zlist, alist = [], []
    n_batches = 50
    print('sampling z vectors from training set')
    for k in range(n_batches):
        cond_, target_, action_ = dataloader.get_batch('train')
        pred, g_pred, z = model(Variable(cond_.cuda()), Variable(target_.cuda()))
        zlist.append(z.data.cpu())
        alist.append(action_)
    zlist = torch.stack(zlist).view(opt.batch_size * n_batches, -1)
    znp = zlist.cpu().numpy()
    # if more than 2D, compute PCA so we can visualize the z distribution
    if zlist.size(1) > 2:
        pca = PCA(n_components=2)
        znp = pca.fit(znp).transform(znp)
    # make and save scatter plot
    plt.scatter(znp[:, 0], znp[:, 1], s=2)
    plt.savefig('{}/z_pca_dist.png'.format(save_dir))
    plt.clf()

    # compute the residuals and predictions with the true z computed using the target
    pred, g_pred, z_true = model(vcond, vtarget)
    err = vtarget - g_pred
    pred = pred.data.view(opt.batch_size, opt.npred, opt.nc, opt.height, opt.width)
    err = err.data.view(opt.batch_size, opt.npred, opt.nc, opt.height, opt.width)

    for b in range(opt.batch_size):
        img = dataloader.plot_seq(cond[b].cpu().unsqueeze(0), pred[b].cpu().unsqueeze(0))
        fname = '{}/ep{}'.format(save_dir, b)
        if not os.path.isdir(fname):
            os.system('mkdir -p ' + fname)
        torchvision.utils.save_image(img, '{}/z_true.png'.format(fname))
        img = dataloader.plot_seq(cond[b].cpu().unsqueeze(0), err[b].cpu().unsqueeze(0))
        torchvision.utils.save_image(img, '{}/residual.png'.format(fname))


    # compute predictions for different z vectors, each prediction has its own folder
    nz = 50
    mov = []
    for indx in range(nz):
        mov.append([])
        print(indx)
        z = Variable(torch.zeros(opt.batch_size, 1, model.opt.n_latent).cuda())
        z.data.copy_(zlist[indx].view(1, model.opt.n_latent).expand(z.size()))
        pred_z = model.decode(vcond, z)
        pred_z = pred_z.data.view(opt.batch_size, opt.npred, opt.nc, opt.height, opt.width)
        for b in range(opt.batch_size):
            img = dataloader.plot_seq(cond[b].cpu().unsqueeze(0), pred_z[b].cpu().unsqueeze(0))
            fname = '{}/ep{}'.format(save_dir, b)
            torchvision.utils.save_image(img, '{}/z{}.png'.format(fname, indx))
            mov[-1].append(img)


    # write in movie form for easier viewing
    for indx in range(nz):
        mov[indx]=torch.stack(mov[indx])
    mov=torch.stack(mov)
    mov=mov.permute(1, 0, 3, 4, 2).cpu().clone()
    for b in range(opt.batch_size):
        imageio.mimwrite('{}/movie{}.mp4'.format(save_dir, b) , mov[b].cpu().numpy() , fps = 5)
