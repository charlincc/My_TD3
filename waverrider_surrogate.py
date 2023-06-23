from keras.layers import Dense
from keras.models import Sequential
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class surrogate_net():

    def load_data(self)->None:

        self.input_data = np.array(pd.read_csv('data/all_input_value.dat', sep='\s+', header=None))  # 输入数据

        self.output_data = np.array(pd.read_csv('data/all_output_value.dat', sep='\s+', header=None))# 输出数据



    def process_data(self)->None:

        self.scaler1 = MinMaxScaler()

        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.input_data,self.output_data,test_size=0.1)#划分测试集，训练集

        self.x_train_ad = self.x_train  #备份一个

        self.x_train = self.scaler1.fit_transform(self.x_train)

        self.x_test = self.scaler1.fit_transform(self.x_test)#首先归一测试集和训练集,验证集
        # self.pre_input = self.pre_input.reshape(( self.pre_input.shape[0],1,self.pre_input.shape[1]))#塑造Lstm

    def build_model(self):

        model = Sequential()
        model.add( Dense( 64,input_dim=self.x_train.shape[1],activation='relu',name='layer0') )
        model.add(Dense( 128,activation='relu',name='layer12'))
        model.add( Dense( 64,activation='relu',name='layer01') )
        model.add(Dense( self.y_train.shape[1], name="out") )
        self.model = model

        return self.model

    def train_model(self) -> None:

        self.model.compile(optimizer=keras.optimizers.RMSprop(1e-6), loss="mse")
        self.history = self.model.fit(self.x_train, self.y_train,validation_split=0.2, epochs=10000)  # fit开始训练
        self.model.save_weights('surrogate_model/surrogate')

    def test_error(self)->None:

        moodel = self.model

        moodel.load_weights('surrogate_model/surrogate')

        res = abs(( moodel.predict(self.x_test) - self.y_test ) / self.y_test * 100 )
        plt.plot(np.linspace(1,res.shape[0],res.shape[0]),res[:,0])
        plt.xlabel('Number of data')
        plt.ylabel('Error value(%)')
        plt.legend(['L/D'])
        plt.show()




agent = surrogate_net()
agent.load_data()
agent.process_data()
agent.build_model()
# try_model.model_train()
# agent.test_error()
