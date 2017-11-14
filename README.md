# Prediction under Uncertainty with Error-Encoding Networks

Code to train the models described in the paper "Prediction under Uncertainty with Error-Encoding Networks". 



## Usage:


### Data:

First you will need the data. The Poke dataset can be downloaded from the author's website:

```
http://ashvin.me/pokebot-website/
```
The Atari and Flappy Bird datasets can be downloaded from the following links:

```
https://drive.google.com/open?id=1E6tH6fctOQhObs004IkyM8UmrTkFqWmx
https://drive.google.com/file/d/1hMuk5zRM5BnUt_3fNTHR7cohxSgiVnBd/view?usp=sharing
```

You will then want to specify the dataset paths in the file config.json. This contains the dataset parameters for each task such as image size, number of frames to condition on, etc. If you want to try the method on a new dataset, just add an entry to that file. The script data_test.py will check if the data has been loaded properfly and will display some images.

### Training

The first step is to train a deterministic network. This can be done with the following script

```
python train_g_network.py -task breakout
```

We will want to initialize the latent variable network with the weights of the deterministic one. 
Once this is trained, you can train the latent variable network with the desired number of latent variables by running:

```
python train_f_network.py -task breakout -n_latent 2
```

This script automatically loads the deterministic model weights from the folder specified with the -save_dir option, so make sure it is the same as in the previous script. 

### Visualization

After training, you can run the script visualize.py which will generate frame predictions for different z vectors. These will be saved in a folder with the same name as the saved model file, just with '.viz' appended to it. By default all the models in the save_dir directory will have generations produced. 


