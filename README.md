# Prediction under Uncertainty with Error-Encoding Networks

Code to train the models described in the paper "Prediction under Uncertainty with Error-Encoding Networks". 



## Usage:


### Data:

We also provide all the datasets in one big file which can be downloaded here:

```
url
```

Then extract it:

```
tar -xvf een_data.tar.gz
```

You can also download the Poke and TORCS datasets from the author's websites:

```
http://ashvin.me/pokebot-website/
```


### Training

The first step is to train a deterministic network. This can be done with the following script

```
python train_g_network.py -task breakout -datapath /path/to/your/data
```

You can change the task option to any of the other tasks. 
Once this is trained, you can train the latent variable network with the desired number of latent variables by running:

```
python train_f_network.py -task breakout -n_latent 2 -datapath /path/to/your/data
```

This script automatically loads the deterministic model weights from the folder specified with the -save_dir option, so make sure it is the same as in the previous script. 

### Visualization

After training, you can run the script visualize.py which will generate frame predictions for different z vectors. 

``` 
python visualize.py -save_dir /path/to/models/
```

This will create a new directory for each model in the folder with the same name as the model file with '.viz' appended to it. This will contain one subfolder per set of conditioning frames, each with several generations using different z vectors. These are also saves as MP4 movie files for easier viewing. 


