import os, random, glob, pdb, math
import cPickle as pickle
import numpy
from scipy import misc
import torch
import torchvision
import utils


class ImageLoader(object):

    def _load_set(self, split):
        print('loading {} set'.format(split))
        datalist = []
        datapath = '{}/{}/'.format(self.arg.get("datapath"), split)
        for fdname in os.listdir(datapath):
            fd_datalist = []
            abs_fdname = os.path.join(datapath, fdname)
            print("loading {}".format(abs_fdname))
            presaved_npy = glob.glob(os.path.join(abs_fdname, "presave.npy"))
            if len(presaved_npy) == 1:
                fd_datalist = numpy.load(presaved_npy[0])
            elif len(presaved_npy) == 0:
                for abs_fname in sorted(glob.glob(os.path.join(abs_fdname, "*.jpg"))[:-1]):
                    print('reading {}'.format(abs_fname))
                    img = misc.imread(abs_fname)
                    r_img = misc.imresize(img, (self.height, self.width))
                    fd_datalist.append(r_img)
                fd_datalist = numpy.transpose(numpy.array(fd_datalist), (0, 3, 1, 2))
                numpy.save(os.path.join(abs_fdname, "presave.npy"), fd_datalist)
            else:
                raise ValueError
            actions = numpy.load(abs_fdname + '/actions.npy')
            datalist.append({'frames': fd_datalist, 'actions': actions})
        return datalist

    def __init__(self, arg):
        super(ImageLoader, self).__init__()
        self.arg = arg
        self.datalist = []

        self.height = arg.get('height')
        self.width = arg.get('width')
        self.nc = arg.get('nc')
        self.ncond = arg.get('ncond', 1)
        self.npred = arg.get('npred', 1)
        self.datalist_train = self._load_set('train')
        self.datalist_test = self._load_set('test')

        # keep some training data for validation
        self.datalist_valid = self.datalist_train[-3:]
        self.datalist_train = self.datalist_train[:-3]
            
        # pointers
        self.iter_video_ptr = 0
        self.iter_sample_ptr = self.ncond
        print("Dataloader constructed done")


    def reset_ptrs(self):
        self.iter_video_ptr = 0
        self.iter_sample_ptr = self.ncond
            
    def _sample_time(self, video, actions, num_cond, num_pred):
        start_pos = random.randint(0, video.shape[0]-2)
        cond_frames = video[start_pos]
        pred_frames = video[start_pos+1]
        actions = actions[start_pos]
        return cond_frames, pred_frames, actions

    def _iterate_time(self, video, start_pos, actions, num_cond, num_pred):
        cond_frames = video[start_pos]
        pred_frames = video[start_pos+1]
        actions = actions[start_pos]
        return cond_frames, pred_frames, actions

        
    def get_batch(self, split):
        if split == 'train':
            datalist = self.datalist_train
        elif split == 'valid':
            datalist = self.datalist_valid
        elif split == 'test':
            datalist = self.datalist_test

        cond_frames, pred_frames, actions = [], [], []
        # rolling
        id = 1
        while id <= self.arg.get("batchsize"):
            sample = random.choice(datalist)
            sample_video = sample.get('frames')
            sample_actions = sample.get('actions')                                 
            selected_cond_frames, selected_pred_frames, selected_actions = self._sample_time(
                    sample_video, sample_actions, self.ncond, self.npred)
            assert(len(selected_actions) > 0)
            cond_frames.append(selected_cond_frames)
            pred_frames.append(selected_pred_frames)
            actions.append(selected_actions)
            id += 1

        # processing on the numpy array level 
        cond_frames = numpy.array(cond_frames, dtype='float') / 255.0
        pred_frames = numpy.array(pred_frames, dtype='float') / 255.0
        actions = numpy.array(actions).squeeze()
        # return tensor
        cond_frames_ts = torch.from_numpy(cond_frames).float().cuda()
        pred_frames_ts = torch.from_numpy(pred_frames).float().cuda()
        actions_ts = torch.from_numpy(actions).float().cuda()
        return cond_frames_ts, pred_frames_ts, actions_ts



    def get_iterated_batch(self, split):
        if self.split == 'train':
            datalist = self.datalist_train
        elif self.split == 'test':
            datalist = self.datalist_test
        cond_frames, pred_frames, actions = [], [], []
        # rolling
        id = 1
        while id <= self.arg.get("batchsize"):
            if self.iter_video_ptr == len(datalist):
                return None, None, None
            sample = self.datalist[self.iter_video_ptr]
            sample_video = sample.get('frames')
            sample_actions = sample.get('actions')                                 
            if self.iter_sample_ptr + self.npred > sample_video.shape[0]:
                self.iter_video_ptr += 1
                self.iter_sample_ptr = self.ncond
            else:
                selected_cond_frames, selected_pred_frames, selected_actions = self._iterate_time(
                    sample_video, self.iter_sample_ptr, sample_actions, self.ncond, self.npred)

                assert(len(selected_actions) > 0)
                cond_frames.append(selected_cond_frames)
                pred_frames.append(selected_pred_frames)
                actions.append(selected_actions)
                id += 1
                self.iter_sample_ptr += 1


        # processing on the numpy array level 
        cond_frames = numpy.array(cond_frames, dtype='float') / 255.0
        pred_frames = numpy.array(pred_frames, dtype='float') / 255.0
        actions = numpy.array(actions).squeeze()

        # return tensor
        cond_frames_ts = torch.from_numpy(cond_frames).cuda()
        pred_frames_ts = torch.from_numpy(pred_frames).cuda()
        actions_ts = torch.from_numpy(actions).cuda()
        return cond_frames_ts, pred_frames_ts, actions_ts

    def plot_seq(self, cond, pred):
        cond_pred = torch.cat((cond, pred), 1)
        cond_pred = cond_pred.view(-1, self.nc, self.height, self.width)
        grid = torchvision.utils.make_grid(cond_pred, self.ncond+self.npred, pad_value=1)
        return grid
