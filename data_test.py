import utils, imp, numpy
import matplotlib.pyplot as plt
# tester for dataloader
parser = argparse.ArgumentParser()
parser.add_argument('-task', type=str, default='poke', help='breakout | seaquest | flappy | poke | driving')
parser.add_argument('-datapath', type=str, default='./data/')
opt = parser.parse_args()

data_config = utils.read_config('config.json').get(opt.task)
data_config['datapath'] = '{}/{}'.format(opt.datapath, data_config['datapath'])
data_config['batchsize'] = 64
ImageLoader=imp.load_source('ImageLoader', 'dataloaders/{}.py'.format(data_config.get('dataloader'))).ImageLoader
dataloader = ImageLoader(data_config)

cond, target, action = dataloader.get_batch('train')

# show some images
N = 3
im=dataloader.plot_seq(cond[0:N].unsqueeze(1), target[0:N].unsqueeze(1))
plt.imshow(numpy.transpose(im.cpu().numpy(), (1, 2, 0)))
plt.show()

