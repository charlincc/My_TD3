import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class surrogate_net():

    def load_data(self):#加载数据集

        input_data = np.array(pd.read_csv('data/OLHS_input_5324'))
        output_data = np.array(pd.read_csv('data/OLHS_output_5324'))

        self.input_data = np.array([])
        self.output_data = np.array([])

        for i in output_data:
            if i[1] < 1 and i[1] > 0:
                self.output_data = np.append(self.output_data,i)
                self.input_data = np.append(self.input_data,input_data[int(i[0]) ] )


        self.input_data = np.reshape(self.input_data,(-1,input_data.shape[1]))[:,1:]

        self.output_data = np.reshape(self.output_data,(-1,output_data.shape[1]))[:,1:]


    def process_data(self):#划分数据集

        self.scalar = MinMaxScaler()    #实例化，归一化器

        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.input_data,self.output_data,test_size=0.1)#划分测试集，训练集

        self.x_train_ad = self.x_train  #备份一个

        self.x_train = self.scalar.fit_transform(self.x_train)

        self.x_test = self.scalar.fit_transform(self.x_test)#首先归一测试集和训练集,验证集


    def build_model(self):
        self.model = Sequential()
        self.model.add( Dense(64,input_dim = self.x_train.shape[1],activation = 'relu' ,name='layer0') )
        self.model.add(Dense(128,activation= 'relu'))
        self.model.add(Dense(64,activation= 'relu'))
        self.model.add(Dense(self.y_train.shape[1],activation= 'relu',name='out'))
        return self.model

    def train_model(self):
        self.model.compile(optimizer = tf.keras.optimizers.Nadam(1e-6),loss='MSE')
        self.history = self.model.fit(self.x_train,self.y_train,validation_split=0.2,epochs = 50)
        self.model.save_weights('surrogate_model/surrogate')


    def test_error(self)->None:

        model = self.model
        model.load_weights('surrogate_model/surrogate')
        y_pred = self.model(self.x_test)

        res = abs(( y_pred - self.y_test ) / self.y_test * 100 )
        x_ax = np.linspace(1,res.shape[0] ,res.shape[0])

        y_ax = res[:,0]

        plt.plot(x_ax,y_ax)

        plt.xlabel('Number of data')
        plt.ylabel('Error value(%)')
        plt.legend(['lift drag coe'])
        plt.show()



agent = surrogate_net()
agent.load_data()
agent.process_data()
agent.build_model()
# agent.train_model()
# agent.test_error()
