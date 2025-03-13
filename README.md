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

### • Generator: A neural network utilizing transposed convolution layers to create lifelike face images from random noise.
### • Discriminator: A convolutional neural network designed to distinguish between real and generated images.


