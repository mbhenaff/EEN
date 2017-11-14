#-*- coding: utf-8 -*-
# by Jake Zhao

import os, random, glob, pdb
import cPickle as pickle
import numpy
from scipy import misc
import torch
import torchvision
from collections import OrderedDict
import pylab
from scipy import misc
import imageio, csv
import matplotlib.pyplot as plt
import time


class ImageLoader(object):
    def __init__(self, arg):
        super(ImageLoader, self).__init__()
        self.arg = arg
        self.datalist = []

        self.h = arg.get('height')
        self.w = arg.get('width')
        self.nc = arg.get('nc')
        self.ncond = arg.get('ncond', 1)
        self.npred = arg.get('npred', 1)

        # reading
        self.fnames = ['trace_0', 
                       'trace_1', 
                       'trace_2', 
                       'trace_3', 
                       'trace_4', 
                       'trace_5', 
                       'trace_6', 
                       'trace_7', 
                       'trace_8', 
                       'trace_9', 
                       'trace_10', 
                       'trace_11']
        self.datalist = []
        total_frames = 0
        for fname in self.fnames:
            abs_fname_action = '{}/labels/{}.csv'.format(arg.get("datapath"), fname)

            with open(abs_fname_action, 'rb') as f:
                reader = csv.reader(f)
                actions = map(tuple, reader)
            actions = numpy.array(actions[1:])


            abs_fname_video = '{}/videos/{}'.format(arg.get("datapath"), fname)
            if os.path.isfile(abs_fname_video + '.npy'):
                print('loading presaved numpy array {}'.format(abs_fname_video + '.npy'))
                frames = numpy.load(abs_fname_video + '.npy')
                total_frames += len(frames)
            else:
                print('loading {}'.format(abs_fname_video + '.avi'))
                vid = imageio.get_reader(abs_fname_video + '.avi',  'ffmpeg')
                frames = []
                for i, im in enumerate(vid):
                    r_img = misc.imresize(im, (self.h, self.w))
                    frames.append(r_img)
                frames = numpy.array(frames)
                numpy.save(abs_fname_video + '.npy', frames)

            self.datalist.append({'frames': frames, 'actions': actions})

        # pointers
        self.iter_video_ptr = 0
        self.iter_sample_ptr = self.ncond
        self.datalist_train = self.datalist[:-4]
        self.datalist_valid = self.datalist[-4:-3]
        self.datalist_test = self.datalist[-3:]

        print("Dataloader constructed done, {} frames".format(total_frames))

    def reset_ptrs(self):
        self.iter_video_ptr = 0
        self.iter_sample_ptr = self.ncond
            
    def _sample_time(self, video, actions, num_cond, num_pred):
        start_pos = random.randint(num_cond+1, video.shape[0]-num_pred-2)
#        actions = actions[start_pos-num_cond:start_pos]
        cond_frames = video[start_pos-num_cond:start_pos]
        pred_frames = video[start_pos:start_pos+num_pred]
        actions = actions[(start_pos-1):(start_pos-1)+num_pred]
#        actions = actions[(start_pos-0):(start_pos-0)+num_pred]
        return cond_frames, pred_frames, actions

    def _iterate_time(self, video, start_pos, actions, num_cond, num_pred):
        cond_frames = video[start_pos-num_cond:start_pos]
        pred_frames = video[start_pos:start_pos+num_pred]
        actions = actions[start_pos:start_pos+num_pred] 
        return cond_frames, pred_frames, actions

    def get_iterated_batch(self, set):
        cond_frames, pred_frames, actions = [], [], []
        assert(set == 'test')
        if set == 'train':
            datalist = self.datalist_train
        elif set == 'valid':
            datalist = self.datalist_valid
        elif set == 'test':
            datalist = self.datalist_test

        # rolling
        id = 1
        while id <= self.arg.get("batchsize"):
            if self.iter_video_ptr == len(datalist):
                return None, None, None
            sample = datalist[self.iter_video_ptr]
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
        actions = numpy.array(actions)

        # return tensor
        cond_frames_ts = torch.from_numpy(numpy.transpose(cond_frames, (0, 1, 4, 2, 3))).float()
        pred_frames_ts = torch.from_numpy(numpy.transpose(pred_frames, (0, 1, 4, 2, 3))).float()
        actions_ts = torch.from_numpy(actions.astype('float')).float()
        actions_ts[:, :, 1][actions_ts[:, :, 1].gt(0)] = 1
        return cond_frames_ts, pred_frames_ts, actions_ts.squeeze()



        
    def get_batch(self, set):
        cond_frames, pred_frames, actions = [], [], []
        if set == 'train':
            datalist = self.datalist_train
        elif set == 'valid':
            datalist = self.datalist_valid
        elif set == 'test':
            datalist = self.datalist_test

        # rolling
        id = 1
        while id <= self.arg.get("batchsize"):
            sample = random.choice(datalist)
            sample_video = sample.get('frames')
            sample_actions = sample.get('actions')                                 
            selected_cond_frames, selected_pred_frames, selected_actions = self._sample_time(
                    sample_video, sample_actions, self.ncond, self.npred)

            if len(selected_actions) == 0:
                pdb.set_trace()
            assert(len(selected_actions) > 0)
            cond_frames.append(selected_cond_frames)
            pred_frames.append(selected_pred_frames)
            actions.append(selected_actions)
            id += 1


        # processing on the numpy array level 
        cond_frames = numpy.array(cond_frames, dtype='float') / 255.0
        pred_frames = numpy.array(pred_frames, dtype='float') / 255.0
        actions = numpy.array(actions)

        # return tensor
        cond_frames_ts = torch.from_numpy(numpy.transpose(cond_frames, (0, 1, 4, 2, 3))).float().cuda()
        pred_frames_ts = torch.from_numpy(numpy.transpose(pred_frames, (0, 1, 4, 2, 3))).float().cuda()
        actions_ts = torch.from_numpy(actions.astype('float')).float().cuda()

        return cond_frames_ts, pred_frames_ts, actions_ts.squeeze()

    def get_paired_batch(self, split):
        # hardcoded
        assert(self.npred == 2)
        assert(self.ncond == 4)
        cond, target, action = self.get_batch(split)
        cond1 = cond
        target1 = target[:, 0].unsqueeze(1)
        action1 = action[:, 0]
        cond2 = torch.cat((cond[:, 1:], target1), 1)
        target2 = target[:, 1].unsqueeze(1)
        action2 = action[:, 1]
        return cond1, target1, action1, cond2, target2, action2
        


    def plot_seq(self, cond, pred):
        cond_pred = torch.cat((cond, pred), 1)
        cond_pred = cond_pred.view(-1, self.nc, self.h, self.w)
        grid = torchvision.utils.make_grid(cond_pred, self.ncond+self.npred, pad_value=1)
        return grid
