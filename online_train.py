import numpy as np
import math
import keras
from game2048.agents import ExpectiMaxAgent as TestAgent
from game2048.expectimax import board_to_move
import random
from game2048.game import Game
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten,BatchNormalization,Input
from keras.layers import Conv2D, MaxPooling2D,concatenate,Activation
from keras.optimizers import SGD,Adam
from keras.utils import to_categorical
from keras.models import load_model
from keras import regularizers

from keras.utils import plot_model
from collections import namedtuple
epoch_size=20000000
batch_size=64
capacity=40000
test_size=1000
DEEP=128
flag=5000
# 生成数据
#生成100张图片，每张图片100*100大小，是3通道的。
modelpath='./k_model/CNN_old_158.h5'
Guide = namedtuple('Guide',('state','action'))
max_d=16

class Guides:

	def __init__(self,capacity):
		self.capacity=capacity
		self.memory=[]
		self.position=0
	def push(self,*args):
	###save a transiton###
		if len(self.memory)<self.capacity:
			self.memory.append(None)
		self.memory[self.position]=Guide(*args)
		self.position=(self.position+1)%self.capacity
	def sample(self,batch_size):
		return random.sample(self.memory,batch_size)
	def ready(self,batch_size):
		return len(self.memory)>batch_size
	def _len_(self):
		return len(self.memory)

class ModelWrapper:
	def __init__(self,model,capacity):
		self.model=model
		self.memory=Guides(capacity)
		self.capacity=capacity
		self.training_step=0
		self.refresh=0
	def predict(self,board):
		p=board.reshape(-1)
		data_1=np.zeros((len(p)))
		for i in range(len(p)):
			if p[i]!=0:
				tmp=int(math.log(p[i],2))
				data_1[i]=tmp
		board=np.reshape(data_1,(4,4))
		board = to_categorical(board,max_d)
		return self.model.predict(np.expand_dims(board,axis=0))

	def move(self,game):
		ohe_board=game.board
		suggest=board_to_move(game.board)
		direction=self.predict(game.board).argmax()
		game.move(direction)
		a=random.randint(0,9)
		# if(ohe_board.max()>=128 or a>=7):
		self.memory.push(ohe_board,suggest)
		# print(game.board)
		return game
	def onehot(self,x,shape):
		p=x.reshape(-1)
		data_1=np.zeros((len(p)))
		for i in range(len(p)):
			if p[i]!=0:
				tmp=int(math.log(p[i],2))
				data_1[i]=tmp
		data=np.reshape(data_1,shape)
		data=to_categorical(data,max_d)
		return data


	def train(self,batch,game):
		from game2048.game import Game
		#if(self.training_step<10):
		if self.memory.ready(test_size) and self.refresh%batch_size==1:	
			guides=self.memory.sample(batch)
			X=[]
			Y=[]
			for guide in guides:
				X.append(guide.state)
				ohe_action=[0]*4
				ohe_action[guide.action]=1
				Y.append(ohe_action)
			X=self.onehot(np.array(X),(batch_size,4,4))
			loss,acc=self.model.train_on_batch(np.array(X), np.array(Y))
			# for test############
			if(self.training_step%500==1):
				guides=self.memory.sample(test_size)
				x_test=[]
				y_test=[]
				for guide in guides:
					x_test.append(guide.state)
					ohe_action=[0]*4
					ohe_action[guide.action]=1
					y_test.append(ohe_action)
				x_test=self.onehot(np.array(x_test),(test_size,4,4))
				print(type(x_test))
				score= model.evaluate(x_test, y_test, verbose=1)
				with open('oldtrainvalue.txt','a') as f:
					f.write("\nmyagent")
					f.write("loss"+str(score[0]))
					f.write("accuracy"+str(score[1]))
			#end test################################
			# if (self.memory._len_==20000-1):
			# 	self.memory=Guides(capacity)
			self.training_step+=1
			game=self.move(game)
			self.refresh+=1
			return game
		else:
			game=self.move(game)
			self.refresh+=1
			return game

# model= load_model(modelpath)
# print("load_model"+modelpath)

inputs=Input((4,4,max_d))
conv=inputs


conv21=(Conv2D(DEEP, (2, 1), kernel_initializer='he_uniform')(conv))
conv12=(Conv2D(DEEP, (1, 2), kernel_initializer='he_uniform')(conv))
conv22=(Conv2D(DEEP, (2, 2), kernel_initializer='he_uniform')(conv))
conv33=(Conv2D(DEEP, (3, 3), kernel_initializer='he_uniform')(conv))
conv44=(Conv2D(DEEP, (4, 4), kernel_initializer='he_uniform')(conv))

hidden=concatenate([Flatten()(conv21), Flatten()(conv12), Flatten()(conv22), Flatten()(conv33), Flatten()(conv44)])
x = BatchNormalization()(hidden)
x = Activation('relu')(x)
print(type(x))
# for width in [512,128]:
# 	x=Dense(256, activation='relu')(x)
# 	x=BatchNormalization()(x)
# 	x=Activation('relu')
x=Dense(1024, kernel_initializer='he_uniform',activation='relu')(x)
x=BatchNormalization()(x)
x=Dense(128, kernel_initializer='he_uniform',activation='relu')(x)
x=BatchNormalization()(x)
outputs=Dense(4,activation='softmax',kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.001))(x)
# outputs=np.argmax()
model=Model(inputs,outputs)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
M1=ModelWrapper(model,capacity)

game = Game(4, 2048)
while M1.training_step<epoch_size:
	# print(game.board)
	if(game.board.max()<=2048 and game.board.min()==0):
		game=M1.train(batch_size,game)
	else:
		game = Game(4, 2048)
		game=M1.train(batch_size,game)
		print("new game",str(game.board.max()))
	if(M1.training_step==flag):
		model.save("./k_model/CNN_old_16_"+str(round(M1.training_step/10000)+200)+".h5")
		flag+=5000
		with open('oldtrainvalue.txt','a') as f:
			f.write("\nnewmodel")
			f.write("\n./k_model/CNN_old_16_"+str(round(M1.training_step/10000)+200)+".h5")
	# print("M1.training_step",M1.training_step)

# model.summary()
# model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch_size,shuffle=True,verbose=2,validation_split=0.3)

# model.train_on_batch(np.array(x_train), np.array(y_train))
# print(np.shape(x_train))
# print(np.argmax(model.predict(np.reshape(x_test[0],(1,4,4,5)))))
# score= model.evaluate(x_test, y_test, verbose=1)
# print(score)
# print('loss:',score[0])
# print('accuracy:',score[1])

