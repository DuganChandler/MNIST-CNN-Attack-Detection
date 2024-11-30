# Federated Learning on MNIST data with attack mitigation/detection
This is a CNN model that has been trained on the MNIST data set in three levels:
- Level 1:
    - Centralized learning, no attack detection/mitigation
- Level 2:
    - Federated Learning, no attack detection/mitigation
- Level 3:
    - Federated Learning with attack detection/mitigation
## How to Run
- Create a python vertual environment with conda or pip
- Check your CUDA version on your system:
```
nvidia-smi
```
- Ensure that you have pytorch install from the [pytorch Official Website](https://pytorch.org/get-started/locally/), copy the command according to CUDA version
for example: 
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```
or 
```
pip install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```
- After pytorch is installed, you will be able to run any of the models via:
```
py level_x.py
```
This will train, evaluate, and save the best model.
