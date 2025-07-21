# Pix2Pix Image-to-Image Translation

## Introduction
This project focuses on **Image-to-Image Translation** using a **Conditional Generative Adversarial Network (cGAN)**, specifically the **pix2pix** model. The dataset used for this project is the [CMP Facades Dataset](http://cmp.felk.cvut.cz/~tylecr1/facade/), which contains images of building facades. The goal is to train a model that can translate architectural label maps into photorealistic images.

## 🫡 You can help me by Donating
[![BuyMeACoffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/heytanix)

### Dataset Overview:
- **Dataset**: CMP Facades
- **Content**: 400 training images, 106 validation images, 100 test images.
- **Image Format**: Each image is a 512x256 composite, with the architectural label map on the left and the corresponding real photo on the right.
- **Classes**: The model learns a mapping between two domains: architectural labels and photos.

**Objective**:
The primary goal of this project is to train a deep learning model to successfully translate an input architectural drawing into a photorealistic image of a building facade, complete with realistic textures, lighting, and details.

## Models & Architectures Used
To address the image translation task, the following architectures were implemented within a cGAN framework:

1.  **Generator (U-Net Architecture)**: An encoder-decoder network with skip connections. The skip connections are vital for passing low-level information (like edges) directly to the decoder, which helps generate high-quality, detailed images.
2.  **Discriminator (PatchGAN Architecture)**: A convolutional classifier that determines if `N x N` patches of an image are real or fake. This encourages the generator to produce sharp, high-frequency details across the entire image.
3.  **Conditional Generative Adversarial Network (cGAN)**: The overarching framework where the generator and discriminator are trained adversarially. The "conditional" aspect means both models receive the input label map as a condition, guiding the image generation process.

## Project Workflow
### Step 1: Data Loading & Exploration
- Loaded the dataset containing composite images.
- Implemented a custom PyTorch `Dataset` class to split each composite image into two separate images: the input (architectural labels) and the target (real photo).

### Step 2: Data Preprocessing
- Applied a series of transformations to augment the data and prepare it for the model.
- Transformations included resizing the images, applying random horizontal flips and crops for data augmentation, and normalizing the pixel values to a range of `[-1, 1]`.

### Step 3: Model Training
- Trained the Generator and Discriminator in an adversarial loop.
- The **Generator** attempts to create realistic photos from the input labels.
- The **Discriminator** attempts to distinguish between the real photos and the ones created by the generator.
- The training process uses a combined loss function: an **Adversarial Loss** (to make images look real) and an **L1 Loss** (to ensure the output is structurally similar to the input).

### Step 4: Evaluation
- After training, the generator was used to translate a set of unseen validation images.
- The quality of the output was evaluated visually by comparing the input label, the generated photo, and the ground truth photo.

### Conclusion
The **pix2pix cGAN** model successfully learned the complex mapping from abstract architectural labels to photorealistic images.

- The **U-Net Generator** was effective at reconstructing the overall structure, while the skip connections helped preserve fine details.
- The **PatchGAN Discriminator** proved crucial for generating sharp, high-quality textures and avoiding the blurriness often seen with other loss functions like L2.
- The combination of **L1 Loss** and **Adversarial Loss** was key to achieving both structural accuracy and photorealism.

## Acknowledgments
- **Original Paper**: [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) by Isola, Zhu, Zhou, and Efros.
- **Dataset**: [CMP Facades Dataset](http://cmp.felk.cvut.cz/~tylecr1/facade/)
- **Framework**: [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
