#-*- coding: utf-8 -*-
# by Jake Zhao

import os, random, glob, pdb
import cPickle as pickle
import numpy
from scipy import misc
import torch
import torchvision
from collections import OrderedDict


class ImageLoader(object):
    def __init__(self, arg):
        super(ImageLoader, self).__init__()
        self.arg = arg
        self.datalist = []

        self.h = arg.get('height')
        self.w = arg.get('width')
        self.nc = arg.get('nc')
        self.ncond = arg.get('ncond', 2)
        self.npred = arg.get('npred', 1)
        self.srate = arg.get('srate', 4)
        self.allow_newpipe = arg.get('allow_newpipe', True)

        # reading
        self.datalist = []
        print(arg.get("datapath"))
        for fdname in os.listdir(arg.get("datapath") + '/data/'):
            fd_datalist = []
            abs_fdname = os.path.join(arg.get("datapath") + '/data/', fdname)
            presaved_npy = glob.glob(os.path.join(abs_fdname, "*.npy"))
            if len(presaved_npy) == 1:
                fd_datalist = numpy.load(presaved_npy[0])
            elif len(presaved_npy) == 0:
                for abs_fname in sorted(glob.glob(os.path.join(abs_fdname, "*.png"))[:-1]):
                    img = misc.imread(abs_fname)
                    r_img = misc.imresize(img, (self.h, self.w))
                    fd_datalist.append(r_img)

                fd_datalist = numpy.transpose(numpy.array(fd_datalist), (0, 3, 1, 2))
                print('saving images as numpy array: {}'.format(os.path.join(abs_fdname, "presave.npy"), fd_datalist))
                numpy.save(os.path.join(abs_fdname, "presave.npy"), fd_datalist)
            else:
                raise ValueError
            self.datalist.append(fd_datalist)

        # split
        self.train_datalist = self.datalist[:-5]
        self.valid_datalist = self.datalist[-5:-3]
        self.test_datalist = self.datalist[-3:]

        # pointers
        self.iter_video_ptr = 0
        self.iter_sample_ptr = self.ncond*self.srate

        print("Dataloder for Flappy bird constructed done")


    def reset_ptrs(self):
        self.iter_video_ptr = 0
        self.iter_sample_ptr = self.ncond*self.srate

    def _parse_file(self, filename):
        with open(filename) as handler:
            event_info = handler.readlines()
            content = OrderedDict()
            for elem in event_info:
                elem = elem.split(',')
                img_id = elem[0].split()[1]
                key = elem[1].split(':')[0].split()[1]
                event = elem[1].split(':')[1].split()[1]
                content[img_id] = [int(key), int(event)]
        return content
            
    def _sample_time(self, video, num_cond, num_pred):
        start_pos = random.randint(num_cond*self.srate,
                video.shape[0]-num_pred*self.srate-1)
        cond_interval = video[start_pos-num_cond*self.srate:start_pos]
        pred_interval = video[start_pos:start_pos+num_pred*self.srate]
        # subsample
        cond_frames = cond_interval[::self.srate]
        pred_frames = pred_interval[::self.srate]
        return cond_frames, pred_frames
    
    def _sample_event_time(self, video, events, num_cond, num_pred):
        start_pos = random.randint(num_cond*self.srate,
                video.shape[0]-num_pred*self.srate-1)
        
        cond_interval = video[start_pos-num_cond*self.srate:start_pos]
        pred_interval = video[start_pos:start_pos+num_pred*self.srate]
        events_interval = events[start_pos-self.srate+1:start_pos+num_pred*self.srate]
        # subsample
        cond_frames = cond_interval[::self.srate]
        pred_frames = pred_interval[::self.srate]
        # not sampling the events; only the prediction events
        events_frames = events_interval[:-(self.srate-1)]
        return cond_frames, pred_frames, events_frames

    def _iterate_time(self, video, pos, events, num_cond, num_pred):
        cond_interval = video[pos-num_cond*self.srate:pos]
        pred_interval = video[pos:pos+num_pred*self.srate]
        events_interval = events[pos-self.srate+1:pos+num_pred*self.srate]
        # subsample
        cond_frames = cond_interval[::self.srate]
        pred_frames = pred_interval[::self.srate]
        # not sampling the events; only the prediction events
        events_frames = events_interval[:-(self.srate-1)]
        return cond_frames, pred_frames, events_frames
    
    def get_batch(self, set):
        cond_frames, pred_frames = [], []
        if set == "train":
            this_set = self.train_datalist
        elif set == "valid":
            this_set = self.valid_datalist
        elif set == "test":
            this_set = self.test_datalist
        else:
            raise ValueError
        # rolling
        id = 1
        while id <= self.arg.get("batchsize"):
            selected_video = random.choice(this_set)
            if (selected_video.shape[0]-self.npred*self.srate-1 > self.ncond*self.srate):
                selected_cond_frames, selected_pred_frames = self._sample_time(
                    selected_video, self.ncond, self.npred)
                if self.allow_newpipe:
                    pass
                elif self.rule_out_new_pipe(selected_cond_frames, selected_pred_frames):
                    continue
                cond_frames.append(selected_cond_frames)
                pred_frames.append(selected_pred_frames)
                id += 1

        # processing on the numpy array level 
        cond_frames = numpy.array(cond_frames, dtype='float') / 255.0
        pred_frames = numpy.array(pred_frames, dtype='float') / 255.0

        # return tensor
        cond_frames_ts = torch.from_numpy(cond_frames).float().cuda()
        pred_frames_ts = torch.from_numpy(pred_frames).float().cuda()
        return cond_frames_ts, pred_frames_ts, None

    def get_event_batch(self, set):
        cond_frames, pred_frames, events_frames = [], [], []

        if set == "train":
            this_set = self.e_train_datalist
            this_set_events = self.e_train_event_datalist
        elif set == "test":
            this_set = self.e_test_datalist
            this_set_events = self.e_test_event_datalist
        else:
            raise ValueError


        # rolling
        id = 1
        while id <= self.arg.get("batchsize"):
            selected_video, selected_video_events = random.choice(
                    zip(this_set, this_set_events))
            if (selected_video.shape[0]-self.npred*self.srate-1 > self.ncond*self.srate):
                selected_cond_frames, selected_pred_frames, selected_events_frames = self._sample_event_time(
                    selected_video, selected_video_events, self.ncond, self.npred)
                if self.allow_newpipe:
                    pass
                elif self.rule_out_new_pipe(selected_cond_frames, selected_pred_frames):
                    continue
                cond_frames.append(selected_cond_frames)
                pred_frames.append(selected_pred_frames)
                events_frames.append(selected_events_frames)
                id += 1

        # processing on the numpy array level 
        cond_frames = numpy.array(cond_frames, dtype='float') / 255.0
        pred_frames = numpy.array(pred_frames, dtype='float') / 255.0

        # return tensor
        cond_frames_ts = torch.Tensor(cond_frames)
        pred_frames_ts = torch.Tensor(pred_frames)
        self.batch_size = self.arg.get("batchsize")
        events_frames_ts = torch.Tensor(numpy.array(events_frames)).view(self.batch_size, self.npred, 4)
        return cond_frames_ts, pred_frames_ts, events_frames_ts.squeeze()

    
    def get_iterated_batch(self, set):
        cond_frames, pred_frames, events_frames = [], [], []
        if set == "train_event":
            this_set = self.e_train_datalist
            this_set_events = self.e_train_event_datalist
        elif set == "test":
            this_set = self.e_test_datalist
            this_set_events = self.e_test_event_datalist
        else:
            raise ValueError

        # iterating through
        id = 1
        while id <= self.arg.get("batchsize"):
            if self.iter_video_ptr == len(this_set):
                return None, None, None
            selected_video = this_set[self.iter_video_ptr]
            selected_video_events = this_set_events[self.iter_video_ptr]
            selected_cond_frames, selected_pred_frames, selected_events_frames = \
                    self._iterate_time(selected_video, self.iter_sample_ptr,
                                       selected_video_events, self.ncond, self.npred)
            # pointers
            self.iter_sample_ptr += 1
            if self.iter_sample_ptr + self.npred*self.srate >= selected_video.shape[0]:
                self.iter_video_ptr += 1
                self.iter_sample_ptr = self.ncond*self.srate
            # filter
            if self.allow_newpipe:
                pass
            elif self.rule_out_new_pipe(selected_cond_frames, selected_pred_frames):
                continue
            cond_frames.append(selected_cond_frames)
            pred_frames.append(selected_pred_frames)
            events_frames.append(selected_events_frames)
            id += 1

        # processing on the numpy array level 
        cond_frames = numpy.array(cond_frames, dtype='float') / 255.0
        pred_frames = numpy.array(pred_frames, dtype='float') / 255.0

        # return tensor
        cond_frames_ts = torch.Tensor(cond_frames)
        pred_frames_ts = torch.Tensor(pred_frames)
        return cond_frames_ts, pred_frames_ts, events_frames

    def reset_ptr(self):
        self.iter_video_ptr = 0
        self.iter_sample_ptr = self.ncond*self.srate

    def plot_seq(self, cond, pred, num_pred=0):
        if num_pred == 0:
            num_pred = self.npred
        cond_pred = torch.cat((cond, pred), 1)
        cond_pred = cond_pred.view(-1, self.nc, self.h, self.w)
        grid = torchvision.utils.make_grid(cond_pred, self.ncond+num_pred)
        return grid
    
    def assign_background(self, val):
        self.background = val

    def rule_out_new_pipe(self, cond, pred):
        mean_c = cond[self.ncond-1][:,:3,:].sum(0).mean(0).tolist()
        mean_p_list = []
        for i in range(self.npred):
            mean_p_list.append(pred[i][:,:3,:].sum(0).mean(0).tolist())
        if mean_c[-1] in self.background:
            for elem in mean_p_list:
                if elem[-1] not in self.background:
                    return True
        return False
