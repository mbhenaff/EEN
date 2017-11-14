import numpy, os, random, glob, pdb, gc
import cPickle as pickle
from scipy import misc
import torch, torchvision
import copy, time

class ImageLoader(object):

    def _load_set(self, split, n_episodes):
        datalist = []
        datapath = '{}/{}'.format(self.arg.get("datapath"), split)
        flist = os.listdir(datapath)
        print('loading {} new episodes for {} set'.format(n_episodes, split))
        for i in range(1, n_episodes):
            if split == 'train':
                fdname = random.choice(flist)
            else:
                fdname = flist[i]
            abs_fdname = '{}/{}'.format(datapath, fdname)
            episode = None
            while (episode == None):
#                print('loading {}'.format(abs_fdname))
                try:
                    episode = numpy.load(abs_fdname)
                except:
                    print('problem loading {}'.format(abs_fdname))
                    break

                states = episode['states']
                actions = episode['actions']
                assert(len(states) == len(actions))
                assert(len(states) > 0)
                datalist.append({'states': states, 'actions': actions})
                episode.close()
                gc.collect()
        return datalist



    def __init__(self, arg):
        super(ImageLoader, self).__init__()
        self.arg = arg
        self.datalist = []

        self.h = arg.get('height')
        self.w = arg.get('width')
        self.nc = arg.get('nc')
        self.ncond = arg.get('ncond', 4)
        self.npred = arg.get('npred', 4)
        self.datalist_train = self._load_set('train', 500)
        self.datalist_valid = self._load_set('valid', 200)
        self.datalist_test = self._load_set('test', 200)

        self.iter_video_ptr = 0
        self.iter_sample_ptr = self.ncond
        self.train_batch_cntr = 0
        
        print("Dataloader for Atari constructed done")


    def reset_ptrs(self):
        self.iter_video_ptr = 0
        self.iter_sample_ptr = self.ncond
            
    def _sample_time(self, video, actions, num_cond, num_pred):
        start_pos = random.randint(num_cond+1, video.shape[0]-num_pred-2)
        cond_frames = video[start_pos-num_cond:start_pos]
        pred_frames = video[start_pos:start_pos+num_pred]
        actions = actions[start_pos:start_pos+num_pred] 
        return cond_frames, pred_frames, actions

    def _iterate_time(self, video, start_pos, actions, num_cond, num_pred):
        cond_frames = video[start_pos-num_cond:start_pos]
        pred_frames = video[start_pos:start_pos+num_pred]
        actions = actions[start_pos:start_pos+num_pred] 
        return cond_frames, pred_frames, actions

        
    def get_batch(self, split):
        cond_frames, pred_frames, actions = [], [], []
        if split == 'train':
            # since the training set is large, we fetch new episodes every so often
            if self.train_batch_cntr == 1000:
                self.datalist_train = self._load_set('train', 500)
                self.train_batch_cntr = 0
            else:
                self.train_batch_cntr += 1
            this_set = self.datalist_train
        elif split == 'valid':
            this_set = self.datalist_valid
        elif split == 'test':
            this_set = self.datalist_test

        # rolling
        id = 1
        while id <= self.arg.get("batchsize"):
            sample = random.choice(this_set)
            sample_video = sample.get('states')
            sample_actions = sample.get('actions')                                 
            if len(sample_actions) > self.ncond + self.npred + 2:
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
        actions = numpy.array(actions, dtype='float')

        # return tensor
        cond_frames_ts = torch.from_numpy(cond_frames).float()
        pred_frames_ts = torch.from_numpy(pred_frames).float()
        actions_ts = torch.from_numpy(actions.squeeze()).float()
        return cond_frames_ts.cuda(), pred_frames_ts.cuda(), actions_ts.cuda()



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
        return cond1.clone(), target1.clone(), action1.clone(), cond2.clone(), target2.clone(), action2.clone()



    def get_iterated_batch(self, split):
        if split == 'train':
            this_set = self.datalist_train
        elif split == 'valid':
            this_set = self.datalist_valid
        elif split == 'test':
            this_set = self.datalist_test

        cond_frames, pred_frames, actions = [], [], []
        # rolling
        id = 1
        while id <= self.arg.get("batchsize"):
            if self.iter_video_ptr == len(this_set):
                return None, None, None
            sample = this_set[self.iter_video_ptr]
            sample_video = sample.get('states')
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
                self.iter_sample_ptr += 10

        # processing on the numpy array level 
        cond_frames = numpy.array(cond_frames, dtype='float') / 255.0
        pred_frames = numpy.array(pred_frames, dtype='float') / 255.0
        actions = numpy.array(actions, dtype='float')

        # return tensor
        cond_frames_ts = torch.from_numpy(cond_frames).float()
        pred_frames_ts = torch.from_numpy(pred_frames).float()
        actions_ts = torch.from_numpy(actions.squeeze()).float()
        return cond_frames_ts.cuda(), pred_frames_ts.cuda(), actions_ts.cuda()




    def plot_seq(self, cond, pred):
        cond_pred = torch.cat((cond, pred), 1)
        cond_pred = cond_pred.view(-1, self.nc, self.h, self.w)
        grid = torchvision.utils.make_grid(cond_pred, self.ncond+self.npred, pad_value=1, normalize=True)
        return grid
