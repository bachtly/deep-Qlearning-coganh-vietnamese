# Quick Start
## Setup
### Clone the github project
```
git clone https://github.com/edwardly1002/deep-Qlearning-coganh-vietnamese
cd deep-Qlearning-coganh-vietnamese
```
### Install Tensorflow 2.5.0
```
pip install tensorflow
```
You may want to install `tensorflow-gpu` to accelerate training. Do not forget to uninstall `tensorflow` and `tensorflow-cpu` if you have already installed any of them. 
```
pip install tensorflow-gpu
```
A checkpoint of our trained model is located at [Google Drive](https://drive.google.com/file/d/1Pjd-TRsRWeNlf3BPV9S8B9PQMBEe13oL/view?usp=sharing). Download it and put it in the folder named `cp`.

## Usage of Functions
### Play with AI player
You can play against the AI player by execute:
```
python vshuman.py
```
You then just need to follow the instruction. Remember that your pieces are labeled 1 and is unchangable. To change your side, you need to refactor the script.
### Observe games between virtual players
You can observe the games between AI player vs AI player (`AIvsAI.py`), Minimax vs Minimax (`MvsM.py`), Minimax vs Random (`Mvsrandom.py`), AI vs random (`vsrandom.py`). For example:
```
python AIvsAI.py
```
## Train a model
Train the model by running `train_zero.py`
```
python train_zero.py
```
There are utilities for training I have created as listed below.
- Evaluate the model throughout a large number of games (100) against a random player by using `vsrandom.py`.
- Evaluate the model throughout games against Minimax (depth 1 to 4) by using `evaluate.py`.
- There are multiple checkpoints during training, you may want to plot the model's efficiency over time. Use `record_vs_random.py` to get the WRRG and WDRRG of checkpoints over checkpoints and then plot them using `plot_WRRD.py`. The result of plotting is saved in folder `docs/images`.

# Model 
We have written a report located in `docs`. It will explain what exactly we are trying to do.

# References: 
- The OOP architecture of implementation is inspired from [CodeLearn](https://codelearn.io/sharing/day-ai-danh-tictactoe-voi-deep-learning).
- Minimax algorithm implemented in `src/Minimax.py` is borrowed from [Github Page](https://github.com/voxuannguyen2001/BTL2_AI)

