
## Face Emotion Recognition
Building ML model to classify human facial expressions from face image,using classical machine learning techniques so that it requires less compute and complexity.
using GCN to enhance input image as preprocessing step,DLTP feature descriptor,PCA to reduce the high-dimensional DLTP features and K-ELM classifier to classify the face expression





## Project Roadmap:
- Preprocessing
- Image enhancement
- Feature extraction
- Dimensionality reduction
- Classification

## Preprocessing

- Resize the image with size 48x48

- Convert each image into a gray scale.

## Image Enhancement
- Using contrast enhancement techniques to enhance the input images,to make a powerful feature

- The GCN is a global contrast enhancement technique that transforms the intensity of pixels using a single transformation. 

- GCN performs contrast normalization, which considers all image pixels' value


## Feature Extraction
- Extract the Dynamic Local Ternary Pattern (DLTP). It is a very powerful feature

- It determines the threshold automatically that depends on the values of pixels

- DLTP dynamically determines the threshold and is used to get the texture information for the input face image

- This texture information will help to get the emotion of face from input image


## Dimensionality Reduction
- Reduce the number of input features using PCA

- High-dimensional features affect the performance of classifiers

- Features from DLTP have high-dimensional and redundant information

## Classification
- Kernel Extreme Learning Machine Classifier (K-ELM) used a kernel instead of many hidden nodes in ELM classifier

- This kernel helped to decrease the training time and computational complexity
