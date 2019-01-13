# 2048-api
A 2048 game api for training supervised learning (imitation learning) 
follow ExpectiMax agent and create your own CNN model

# Code structure
* [`game2048/`](game2048/): the main package.
    * [`game.py`](game2048/game.py): the core 2048 `Game` class.
    * [`agents.py`](game2048/agents.py): the `Agent` class with instances.
    * [`displays.py`](game2048/displays.py): the `Display` class with instances, to show the `Game` state.
    * [`expectimax/`](game2048/expectimax): a powerful ExpectiMax agent by [here](https://github.com/nneonneo/2048-ai).
* [`explore.ipynb`](explore.ipynb): introduce how to use the `Agent`, `Display` and `Game`.
* [`static/`](static/): frontend assets (based on Vue.js) for web app.
* [`webapp.py`](webapp.py): run the web app (backend) demo.
* [`evaluate.py`](evaluate.py): evaluate your self-defined agent. test time = 50 
* [`online_train.py`](online_train.py): get your own agent's weight.The structure of the model can be found
* [`CNN_new_141.zip`](CNN_new_141.zip): best model weight file trained by me
# Requirements
* code only tested on linux system (ubuntu 16.04)
* Python 3 (Anaconda 3.6.3 specifically) with numpy and flask
* Tensorflow,keras,numpy
# model structure
![model](preview2048.gif)

# for train
```bash
python online_train.py
```
* you will get your own model taught by ExpectiMax agent

# My own agents
In file ./game2048/agent.py
```python

class MyAgent(Agent):
    def __init__(self, game,display=None):
        super().__init__(game, display)
        self.model1= model_my
        # self.model2= model2
        # self.model3= model3

        # print("load_model",modelpath1)


    def step(self):
    ...
        
        return direction

```

# To compile the pre-defined ExpectiMax agent

```bash
cd game2048/expectimax
bash configure
make
```
# For test your model value

```bash
python evaluate.py
```
you will get the average score of your agent (original test time is 50)

# To run the web app
```bash
python webapp.py
```
![demo](preview2048.gif)

# LICENSE
The code is under Apache-2.0 License.

# For EE369 students from SJTU only
Please read [here](EE369.md).
