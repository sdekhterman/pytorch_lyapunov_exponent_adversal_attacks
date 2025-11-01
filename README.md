# Lyapunov Exponent Informed Adversarial Attacks
 This repository replicates some of the results from, ["Finite-Time Lyapunov Exponents of Deep Neural Networks" from Physical Review Letters 132, 057301 by L. Storm, H. Linander,  J. Bec, K. Gustavsson, and B. Mehlig (2024)](https://doi.org/10.1103/PhysRevLett.132.057301).

This was done to check that the computation of the Finite-Time Lyapunov Exponents (FTLEs) was implemented correctly. 

As of writing the author plans to re-implement more parts of the Storm2024 paper, add adversalial attacks, and investidate if the FTLEs can be used to make better adversarial attacks.  

## Installation
I assumed that the Python virtual enviroment (used to keep python packages version need for the code to run from conflicting with what you have already installed) was created using the conda package. For install instuctions of conda see https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html. 

This repositry also used PyTorch to train the deep neural network. If you have conflicts when installing the requirments.txt consider visiting https://pytorch.org/get-started/locally/ to get a version appropriate for your machine/ container.

After installing conda, in a terminal copy the following set of commands 
```
conda create --name lyap_exp_attacks
conda activate lyap_exp_attacks
git clone https://github.com/sdekhterman/pytorch_lyapunov_exponent_adversal_attacks.git
cd pytorch_lyapunov_exponent_adversal_attacks
pip install -r requirements.txt
cd code
python3 mnist_classification.py
```
