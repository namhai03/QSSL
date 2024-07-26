# Self-supervised Learning with Quantum Classifiers

**Authors:** Minh-Hieu Tran†, Nam-Hai Nguyen†, Phuong-Nam Nguyen, Son Hong Ngo  
**Institution:** Faculty of Computer Science, Phenikaa University, Hanoi, Vietnam  
**Contact:** [nam.nguyenphuong@phenikaa-uni.edu.vn](mailto:nam.nguyenphuong@phenikaa-uni.edu.vn)

## Introduction

This repository contains the implementation of our research on self-supervised learning (SSL) using quantum classifiers, as described in the paper "Self-supervision with Quantum Classifiers". Our work proposes an innovative SSL framework that leverages quantum classifiers to enhance representation learning and overcome the challenge of representation collapse in conventional SSL methods. We demonstrate the effectiveness of our approach using image classification tasks on datasets such as MNIST, KMNIST, and FashionMNIST.

## Key Contributions

- **Quantum Classifiers in SSL:** We introduce a hybrid SSL framework that combines classical convolutional neural networks with quantum classifiers to embed data into the Hilbert space. This approach helps in achieving non-trivial, well-stratified representations.
- **Avoidance of Representation Collapse:** Our method minimizes the Kullback-Leibler (KL) divergence to maximize agreement among positive pairs and minimize agreement among negative pairs, thus avoiding the common issue of representation collapse.
- **Superior Performance:** The proposed framework demonstrates higher predictive accuracy compared to classical classifiers, with up to 65% accuracy on MNIST and over 70% on KMNIST and FashionMNIST without fine-tuning.

## Overview
The framework utilizes a classical convolutional encoder followed by a quantum classifier. The classical encoder has a minimal design with a single convolutional layer suitable for Noisy Intermediate-scale Quantum (NISQ) algorithms. Quantum Convolutional Neural Networks (QCNNs) are then employed to encode features into the quantum domain, allowing for complex decision boundaries and better data stratification.  
![Architecture](https://github.com/namhai03/QSSL/blob/main/images/architecture.png)

## Results
![Results](https://github.com/namhai03/QSSL/blob/main/images/results.png)

## How to use
1. Clone this repo:   ```git clone https://github.com/namhai03/QSSL.git```
2. Requirement:   ```pip install -r requirements.txt```
3. 
