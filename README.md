# DCGAN_CELEBA

## DCGAN - DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORK

This repository features a Deep Convolutional Generative Adversarial Network (DCGAN) designed to generate realistic face images using the CelebA dataset. The project is built with PyTorch and offers support for both training and fine-tuning the GAN model.

## TABLE OF CONTENTS

• Installation

• Dataset

• Model Architecture

• Training

• Fine-Tuning

• Generating Images

• Results

• Acknowledgments

## INSTALLATION

Make sure Python and PyTorch are installed with CUDA support (if applicable). Next, install the necessary dependencies:

pip install torch torchvision tqdm matplotlib

## DATASET

The GAN model is trained using the CelebA dataset, which should be stored as a ZIP file named celeba.zip within the base_dir directory. If the dataset hasn't been extracted yet, the script will handle the extraction automatically.

## MODEL ARCHITECTURE

### • Generator: 
A neural network utilizing transposed convolution layers to create lifelike face images from random noise.
### • Discriminator: 
A convolutional neural network designed to distinguish between real and generated images.

## TRAINING

To train the DCGAN model from scratch, run the script:

python train.py

The training process involves:

• Setting up the models
• Employing Binary Cross Entropy (BCE) loss
• Optimizing with the Adam optimizer 
• Training runs for multiple epochs (default: 3) 

The output directory will store the generated images and trained models (generator.pth, discriminator.pth).

## FINE TUNNING

To fine-tune a pre-trained model for additional epochs, run:

python fine_tune.py

This script loads the pre-trained model weights and continues training the network for an additional 10 epochs by default.

## GENERATING IMAGES

To generate new face images using the trained generator:

python generate.py

The script creates and stores sample images in the designated folder, and also displays them using Matplotlib.

## RESULTS

The output and generated directories will store the created images. As the model undergoes more epochs of training, the image quality progressively enhances.

## ACKNOWLEDGMENTS

• This implementation is based on the DCGAN paper: Radford et al., "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"

• Dataset: CelebA - Large-scale CelebFaces Attributes Dataset
