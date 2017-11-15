# Prediction under Uncertainty with Error-Encoding Networks

Code to train the models described in the paper "Prediction under Uncertainty with Error-Encoding Networks". 



## Usage:


### Data:

First you will need the data. The Poke dataset can be downloaded from the author's websites:

```
http://ashvin.me/pokebot-website/
```

We also provide all the datasets in one big file which can be downloaded here:

```
url
```

Untar the file and modify the paths in train_g_model.py and train_f_model.py to point to the directory where the data is stored.

### Training

The first step is to train a deterministic network. This can be done with the following script

```
python train_g_network.py -task breakout
```

You can change the task option to any of the other tasks. 
Once this is trained, you can train the latent variable network with the desired number of latent variables by running:

```
python train_f_network.py -task breakout -n_latent 2
```

This script automatically loads the deterministic model weights from the folder specified with the -save_dir option, so make sure it is the same as in the previous script. 

### Visualization

After training, you can run the script visualize.py which will generate frame predictions for different z vectors. These will be saved in a folder with the same name as the saved model file, just with '.viz' appended to it. By default all the models in the save_dir directory will have generations produced. 


