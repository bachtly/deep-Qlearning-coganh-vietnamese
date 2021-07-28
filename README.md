*Quick Start
*Setup
In your project folder, execute these commands in cmd window:
```
git clone https://github.com/edwardly1002/deep-Qlearning-coganh-vietnamese
cd deep-Qlearning-coganh-vietnamese
```
To run files of this project, you need to install tensorflow:
```
pip install tensorflow
```
When you need to train a model, you may want to install tensorflow-gpu for faster training:
```
pip install tensorflow-gpu
```



If you want to evaluate my checkpoint, download it from https://drive.google.com/drive/folders/150CQ7dMxiJuHdhvLC0Ztm163cBilj6ea?usp=sharing and modify line 20 in evaluate.py
1. To train model, modify code in train.py
2. To let AI player play vs random or human or minimax player, run vsrandom.py, vshuman.py, evaluate.py respectively. 
3. To modify the environment (which is likely to change the mechanism of the game), do it in src/CoganhEnv.py
4. Move_gen functionality is unneccessary. However, if you wish, it is used for reducing training time. Read the docs/report.pdf for more details

*References: 
- The OOP structure of implementation is inspired from [CodeLearn](https://codelearn.io/sharing/day-ai-danh-tictactoe-voi-deep-learning).
- Minimax algorithm implemented in [Minimax Implementation](src/Minimax.py) is borrowed from [Github Page](https://github.com/voxuannguyen2001/BTL2_AI)

