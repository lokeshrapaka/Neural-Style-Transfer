COMPANY : CODETECH IT SOLUTIONS

NAME : RAPAKA LOKESH

INTERN ID : CT06DN379

DOMAIN : ARTIFICIAL INTELLIGENCE

DURATION : 6 WEEKS

MENTOR : NEELA SANTOSH

# Neural-Style-Transfer

**Overview**

This repository contains the implementation of a Neural Style Transfer (NST) project developed as part of the AI Internship at Codetech IT Solutions. The goal of this task is to apply the visual style of one image (usually an artwork or painting) to the content of another image (such as a photograph). This project demonstrates how deep learning and convolutional neural networks (CNNs) can be used for creative tasks, bridging the gap between technology and art.

Neural Style Transfer is an exciting application of deep learning in the domain of computer vision. It leverages pretrained models to extract and recombine the content features of one image with the style features of another, producing visually compelling output that mimics the original art style while preserving the structure of the base image. The technique gained popularity following the work of Gatys et al. in 2015 and has since evolved into a foundation for AI-assisted design and media applications.

**Objective**

The main objective of this task is to:

Implement a neural style transfer algorithm using deep learning.

Take two images as input: one content image and one style image.

Generate a third image that preserves the structure of the content image while reflecting the artistic style of the second.

Use a pre-trained model (such as VGG19) to extract and recombine content and style features.

The deliverable is a Python script or Jupyter Notebook that demonstrates the process and outputs stylized images.

Tools and Technologies Used
This project utilizes the following tools and libraries:

Python (3.x)

PyTorch or TensorFlow (either framework can be used for VGG-based models)

Matplotlib and PIL for image display and manipulation

torchvision.models.vgg19 for feature extraction layers

NumPy for numerical operations

**Implementation Details
**The NST process involves:

1.Loading the content and style images using PIL or OpenCV.

2.Preprocessing the images to match the model’s input specifications.

3.Passing both images through a pretrained CNN (typically VGG19).

4.Extracting content and style features from specific layers of the network.

5.Optimizing a generated image to minimize:

  Content loss: difference between generated and content image features.
  
  Style loss: difference in Gram matrices (correlation of features) between the generated and style images.

6.Iteratively updating the generated image using gradient descent until the losses are minimized and the output looks visually satisfactory.

The result is a hybrid image that carries the detailed structure of the content image, painted in the brushstrokes or texture patterns of the style image.

**Sample Usage**

To run the style transfer:

    python neural_style_transfer.py --content path/to/photo.jpg --style path/to/artwork.jpg
    
The output image will be saved in the output directory, and intermediate results may be visualized during optimization.

