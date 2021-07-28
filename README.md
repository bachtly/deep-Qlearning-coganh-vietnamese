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
### Observe games between virtual player
You can observe the games between AI player vs AI player (`AIvsAI.py`), Minimax vs Minimax (`MvsM.py`), Minimax vs Random (`Mvsrandom.py`), AI vs random (`vsrandom.py`). For example:
```
python AIvsAI.py
```

If you want to evaluate my checkpoint, download it from https://drive.google.com/drive/folders/150CQ7dMxiJuHdhvLC0Ztm163cBilj6ea?usp=sharing and modify line 20 in evaluate.py
1. To train model, modify code in train.py
2. To let AI player play vs random or human or minimax player, run vsrandom.py, vshuman.py, evaluate.py respectively. 
3. To modify the environment (which is likely to change the mechanism of the game), do it in src/CoganhEnv.py
4. Move_gen functionality is unneccessary. However, if you wish, it is used for reducing training time. Read the docs/report.pdf for more details

*References: 
- The OOP structure of implementation is inspired from [CodeLearn](https://codelearn.io/sharing/day-ai-danh-tictactoe-voi-deep-learning).
- Minimax algorithm implemented in [Minimax Implementation](src/Minimax.py) is borrowed from [Github Page](https://github.com/voxuannguyen2001/BTL2_AI)

