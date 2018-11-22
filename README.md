## Imprving the Discriminator by intorducing  Vatiational Bottleneck Layer (VDB)

## Paper - https://arxiv.org/pdf/1810.00821.pdf

## Dependencies
python>=3.5 / python>=2.7
tensorflow>=1.9

## Data
Have tested with mnsit data set. It will be automatically downloaded to data folder

## Usage
python train-WGAN-GP-VDB.py

## Important Implementations
- Item Generator and Discriminator Architectures are inside model.py script 
- Item Normal wgan-gp loss and discriminator bottleneck layer loss are insode the vdb_losses.py script

## Important Paramters

-Item I_c - This is the information contrain. This is a hyper paramter
-Item Bottleneck Layer Dimentions 
-Item Alpha - This paramters is to update the adaptive lagrange paramters (Documentation can be found inside the code)

## Training Results 
-Item  Trained paramters can be found inside  train-WGAN-GP-VDB.py script 


## Future Work 
-Item The modified descriminator can be easily use with GAIL.
-Item Cheking the effect of varios hyper paramters and how generator behaves acording to them
