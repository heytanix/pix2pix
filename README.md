# pix2pix
Image-to-Image translation with cGAN

## Introduction
This project is a PyTorch implementation of the **pix2pix** model, a Conditional Generative Adversarial Network (cGAN) for image-to-image translation. It's trained on the [CMP Facades Dataset](https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/) to learn how to translate architectural labels into realistic building photos.

## 🫡 You can help me by Donating
![BuyMeACoffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)

## How It Works
The pix2pix model features two main components trained in an adversarial process:

1.  **Generator (U-Net)**: The generator's goal is to create a realistic output image from an input label map. It uses a **U-Net architecture**, which is an encoder-decoder network with skip connections. These connections are crucial because they allow low-level information like edges to flow directly from the encoder to the decoder, helping create sharp, detailed images.

2.  **Discriminator (PatchGAN)**: The discriminator's job is to tell the difference between real image pairs (input label + real photo) and fake pairs (input label + generated photo). Instead of classifying the entire image, the **PatchGAN** discriminator classifies `N x N` patches of the image as real or fake. This approach encourages the generator to produce high-frequency details and penalizes unrealistic artifacts locally.

The model is trained by optimizing a combined loss function that includes both an **Adversarial Loss** and an **L1 Loss** to ensure the generated images are both realistic and structurally accurate.

## Key Features
* **Framework**: Implemented entirely in **PyTorch**.
* **Generator**: U-Net with skip connections to preserve details.
* **Discriminator**: PatchGAN for effective, high-frequency detail generation.
* **Data Handling**: Uses custom PyTorch `Dataset` and `DataLoader` for an efficient data pipeline.
* **Training**: Includes a complete training loop with combined L1 and adversarial loss for stable results.

# Project Workflow
### 1. Setup and Imports
First, we'll import all the necessary libraries. We'll use `torch` and `torchvision` for building the model and handling data, `PIL` for image manipulation, `matplotlib` for plotting, and `numpy`. We also set up the device to use a GPU if one is available.

### 2. Prepare Dataset from Local Path
This step points to your local dataset directory. We define a custom `Dataset` class to handle loading the data. The class assumes your directory contains `train` and `val`/`test` subfolders. Each image in these folders should be a composite of an architectural label map (left side) and a corresponding photo (right side).

Our dataset class will:
1.  Load the composite image from your specified path.
2.  Split it into the input image (label map) and the target image (photo).
3.  Apply data augmentation and normalization transformations.

### 3. Define the Generator (U-Net)
The generator is a **U-Net**, which is an encoder-decoder architecture with skip connections. The encoder downsamples the image to extract features, while the decoder upsamples it to construct the output image. The skip connections pass features from the encoder layers directly to the corresponding decoder layers, helping to preserve low-level details.

### 4. Define the Discriminator (PatchGAN)
The discriminator is a convolutional **PatchGAN** classifier. It takes both the input image and the target (or generated) image, concatenated together, and determines whether `30x30` overlapping patches of the image are real or fake. This structure encourages the generator to create realistic high-frequency details across the entire image.

### 5. Initialize Models, Losses, and Optimizers
Here we initialize the models and move them to the selected device. We define the loss functions: `BCEWithLogitsLoss` for the adversarial component and `L1Loss` for the pixel-wise reconstruction. `L1` loss encourages the generated image to be structurally similar to the target. We also set up Adam optimizers for both networks.

### 6. The Training Loop
This is the core of the GAN. For each batch of images:
1.  **Train the Discriminator**: Calculate the loss for real images and fake images separately. The total loss is the average of the two. Backpropagate and update the discriminator's weights.
2.  **Train the Generator**: Calculate the adversarial loss (how well it fools the discriminator) and the L1 reconstruction loss. The total generator loss is a weighted sum of these two losses. Backpropagate and update the generator's weights.

### 7. Generate and Display Final Output
After training, we use the generator to translate a few images from our test set. We switch the generator to evaluation mode (`gen.eval()`) and visualize the results. The output shows the **Input** architectural map, the **Generated** photo, and the **Ground Truth** photo side-by-side. This allows for a qualitative assessment of the model's performance.
