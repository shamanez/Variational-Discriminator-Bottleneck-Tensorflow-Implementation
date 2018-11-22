## Imprving the Discriminator by intorducing  Vatiational Bottleneck Layer (VDB)

## Paper - https://arxiv.org/pdf/1810.00821.pdf

## Dependencies
- python>=2.7

- tensorflow>=1.9

## Data
Have tested with mnsit data set. It will be automatically downloaded to data folder

## Usage
python train-WGAN-GP-VDB.py

## Important Implementations
- Generator and Discriminator Architectures are inside model.py script 
- Normal wgan-gp loss and discriminator bottleneck layer loss are insode the vdb_losses.py script

## Important Paramters

- I_c - This is the information contrain. This is a hyper paramter
- Bottleneck Layer Dimentions 
- Alpha - This paramters is to update the adaptive lagrange paramters (Documentation can be found inside the code)

## Training Results 
- Trained paramters can be found inside  train-WGAN-GP-VDB.py script 
- There are pre-trained checkpoints in the checkpoints folder
- The images generated during the training progress are insode the sample_images_while_training folder
- Tensorbored Visualizations can be find inside summeries folder

<h3> Results: </h3>
<p align="center">
<img alt="Loss Plot" src="https://github.com/shamanez/Variational-Discriminator-Bottleneck-Tensorflow-Implementation/blob/master/Results.gif"
     width=50% height=50% />
</p>
<br>

## Tensorbored Visualizations 
- You can examine the learning progress by visualizing two loss functions of generator and discriminator 
- Also its impotant to undertand the change in beta parameter(Eq(6) in the paper) with adaptive update method. Here we maximize the beta

<h3> Visualizations while Training: </h3>
<img alt="Loss Plot" src="https://github.com/shamanez/Variational-Discriminator-Bottleneck-Tensorflow-Implementation/blob/master/Results.gif"
     width=50% height=50% />
</p>
<br>

## Future Work 
- The modified descriminator can be easily use with GAIL.
- Cheking the effect of varios hyper paramters and how generator behaves acording to them

