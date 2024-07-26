# Self-supervised Learning with Quantum Classifiers
This repository contains the source code for the paper titled **"Self-supervision with Quantum Classifiers"** by Minh-Hieu Tran, Nam-Hai Nguyen, Phuong-Nam Nguyen, and Son Hong Ngo from the Faculty of Computer Science at Phenikaa University.

## Abstract

Self-supervised learning (SSL) has emerged as a novel method for representation learning from inputs without data annotations. A key challenge in SSL is avoiding the collapse of representations, which leads to poor predictive models. This research proposes an SSL framework that leverages quantum classifiers to address this issue. We achieve well stratified, non-trivial representations by embedding data into the Hilbert space via quantum classifiers. The algorithm works by minimizing the Kullback-Leibler (KL) divergence to maximize the agreement of positive pairs and minimize the agreement of negative pairs. We demonstrate the proof of concept in image classification tasks using MNIST, KMNIST, and FashionMNIST datasets. Classifiers trained with our framework achieve higher predictive performance than classical classifiers, with a maximum accuracy of 65% on MNIST and above 70% on KMNIST and FashionMNIST using unseen samples without fine-tuning.

## Repository Structure

- `code/model.py`: Our model implementation and a classical simple Convolutional Neural Network
- `code/SSL.py`: MemoryBank and our Loss function (Kullback-Leibler Loss)
- `code/train.py`: Training session.
- `code/eval.py`: Evaluating  the accuracy using different split sizes of the evaluated set.
- `code/main_v1.py`: Running experiment with train and eval
- `code/demo.py`: A demo script to run the quantum SSL model with specified arguments.
- `code/main_qssl.ipynb`: A demo script can be run on Google Colab
- `images/architecture.png`: Our proposed framework
- `images/results.png`: Our results on MNIST, KMNIST, and FashionMNIST datasets. 

## Prerequisites
- Python 3.10 or higher
- Required Python packages (listed in `requirements.txt`)

## Installation
Clone the repository and install the required packages:
```sh
git clone https://github.com/namhai03/QSSL.git
cd QSSL
pip install -r requirements.txt
```
## Usage
### Running the Demo
To run the `demo.py`script, you can specify the following command-line arguments:
- **--device**: Device to run on (`cpu` or `cuda:0`). Default is `cuda:0`.
- **--classifier**: Classifier type (`classical` or `quantum`). Default is `quantum`.
- **--seed**: Seed. Default is a random seed.
- **--nlayers**: Number of layers. Default is `2`.
- **--gamma**: Gamma value in range (0, 1). Default is `0.5`.
- **--data**: Dataset to use (`MNIST`, `FashionMNIST`, `KMNIST`). Default is `MNIST`.
- **--c1**: Class label 1 (0-9). Default is `0`.
- **--c2**: Class label 2 (0-9), different from c1. Default is `1`.

**Run the `demo.py` with default settings:**
```sh
python demo.py
```

**Run the script with custom settings:**
```sh
python demo.py --device cuda:0 --seed 42 --classifier classical --nlayers 3 --gamma 0.7 --data FashionMNIST --c1 0 --c2 1
```

**Run on `Google Colab`:**
- Download the `code/main_qssl.ipynb` file
- Upload it on Google Colab, and run cell by cell.

### Argument Validation
- **device**: Must be either `cpu` or `cuda:0`
- **Gamma**: Must be within the range (0, 1).
- **Class labels**: c1 and c2 must be between 0 and 9, and they must not be equal.

If any argument is invalid, an appropriate error message will be displayed.

## Project Description
### Self-Supervised Learning with Quantum Classifiers
This project introduces a novel SSL framework using quantum classifiers embedded in the Hilbert space. Our approach addresses the collapse of representations in contrastive learning by leveraging quantum properties, enabling better metric learning and enhancing the predictive power of the models.

### Proposed Framework
The framework utilizes a classical convolutional encoder followed by a quantum classifier. The classical encoder has a minimal design with a single convolutional layer suitable for Noisy Intermediate-scale Quantum (NISQ) algorithms. Quantum Convolutional Neural Networks (QCNNs) are then employed to encode features into the quantum domain, allowing for complex decision boundaries and better data stratification.  
![Architecture](https://github.com/namhai03/QSSL/blob/main/images/architecture.png)

### Experimental Results
The framework is evaluated on three datasets: MNIST, KMNIST, and FashionMNIST. Quantum classifiers demonstrate superior performance compared to classical counterparts, achieving higher accuracy with fewer parameters.
![Results](https://github.com/namhai03/QSSL/blob/main/images/results.png)

### Key Contributions
- **Quantum Classifiers in SSL:** We introduce a hybrid SSL framework that combines classical convolutional neural networks with quantum classifiers to embed data into the Hilbert space. This approach helps in achieving non-trivial, well-stratified representations.
- **Avoidance of Representation Collapse:** Our method minimizes the Kullback-Leibler (KL) divergence to maximize agreement among positive pairs and minimize agreement among negative pairs, thus avoiding the common issue of representation collapse.
- **Superior Performance:** The proposed framework demonstrates higher predictive accuracy compared to classical classifiers, with up to 65% accuracy on MNIST and over 70% on KMNIST and FashionMNIST without fine-tuning.

## Citation
If you find this work useful, please cite the following paper:
```BibTeX
@article{tran2024quantumssl,
  title={Self-supervision with Quantum Classifiers},
  author={Minh-Hieu Tran and Nam-Hai Nguyen and Phuong-Nam Nguyen and Son Hong Ngo},
  journal={...},
  year={2024}
}
```
## Contact 
For any questions or inquiries, please contact the corresponding author at 
[nam.nguyenphuong@phenikaa-uni.edu.vn](mailto:nam.nguyenphuong@phenikaa-uni.edu.vn)

