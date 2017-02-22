# ML@B BOOTCAMP MATERIAL 

This is where all Machine Learning at Berkeley's Bootcamp material is housed. Feel free to use for pedagogical purposes!

Slides are available on this google drive https://drive.google.com/drive/folders/0B_WdXIV-ueK1eTh6Z19NQ01MUjg

Table of Contents 
====
1. Perceptron  
  
  1.1. Binary Classification Problem  
  1.2. Risk Function  
  1.3. Gradient Descent  
  1.4. Perceptron Algorithm
2. SVM  
  
  2.1. Hard Margin SVM  
  2.2. Soft Margin SVM  
3. Decision Trees + Neural Networks  
  
  3.1. Overview 
  
  3.2.  Information Gain / Entropy
  
  3.3. Decision Tree Algorithm
  
  3.4. Random Forrests
  
  3.5. Biological Motivation for Neural Nets  
  
  3.6.Intro to Neural Nets
  
  3.7. Activation Functions
  
  3.8. Feed Forward
  
  3.9. Backpropagation
  
  3.10. Neural Nets Algorithm
  
4. Data Science and Unsupervised Models 

  4.1. K-means clustering  
  
  4.2. Hierarchical Clustering  
  
  4.3. Spectral Clustering  
  
  4.4. Dimensionality Reduction: PCA  
  
  4.5. Dimensionality Reduction: SVD
  
  4.6. Data Visualization
  
5. Deep Learning
   
   5.1. Tensorflow and Keras
   5.2. More advanced Neural Nets
6. AI and Reinforcement Learning 
  6.1. Potpourri for Neural Networks

# Docker Quickstart

This is a quickstart guide to get you up and running. There is a more comprehensive guide for
jupyter notebooks [here](https://github.com/kaggledecal/sp17/blob/master/DockerCheatsheet.md).

## Starting a jupyter container
**Start a jupyter notebook container without mounting a directory**

`docker run -d -p 8888:8888 jupyter/scipy-notebook`

**Start a jupyter container with your current directory mounted**

`docker run -d -p 8888:8888 -v "$(pwd)":/home/jovyan/work jupyter/scipy-notebook`

*Notes*
* Do not change `/home/jovyan/work`. This is a setting for the container


Most of the material was taken from Berkeley's CS189 course and adapted from [last year's bootcamp](https://github.com/jpark96/ml-b-bootcamp-public). Many thanks to Professor Jonathan Shewchuck, who taught the course during the Spring 2016! Here's the reference to his notes (I VERY MUCH suggest this for a stronger mathematical understanding behind the algorithms): https://people.eecs.berkeley.edu/~jrs/papers/machlearn.pdf

## Setting up your environment
If you have not yet installed jupyter on your machine, you'll probably get an error message when you run the above command. 
### Mac OS 10.10+, Linux, and Windows 10
Use [Docker](https://docs.docker.com/engine/installation/).
### Windows < 10 and Mac OS < 10.10
[Anaconda](https://www.continuum.io/downloads).
### What should I use?
Choosing either package is to your discretion; we encourage Docker because it will allow you to quickly prototype papers like [Style Transfer](https://hub.docker.com/r/kchentw/neural-style/), ([paper](http://arxiv.org/abs/1508.06576)).

