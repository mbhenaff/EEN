import torch
import os, json
from datetime import datetime

def one_hot(x, d):
    n = x.ndimension()
    x = x.long().cpu()
    if n == 1:
        x_onehot = torch.zeros(x.size(0), d)
        x_onehot.scatter_(1, x.unsqueeze(1), 1)
    elif n == 2:
        x_onehot = torch.zeros(x.size(0), x.size(1), d)
        x_onehot.scatter_(2, x.unsqueeze(2), 1)
    elif n == 3:
        x_onehot = torch.zeros(x.size(0), x.size(1), x.size(2), d)
        x_onehot.scatter_(3, x.unsqueeze(3), 1)

    return x_onehot.squeeze()



# Logging function
def log(fname, s):
    if not os.path.isdir(os.path.dirname(fname)):
            os.system("mkdir -p " + os.path.dirname(fname))
    f = open(fname, 'a')
    f.write(str(datetime.now()) + ': ' + s + '\n')
    f.close()


def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object

