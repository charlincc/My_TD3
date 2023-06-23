from waverrider_surrogate import agent
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler


class MY_ENV:

    def __init__(self):

        self.model = agent.build_model()

        self.model.load_weights('surrogate_model/surrogate')

        self.cnt_step = 0 #初始化步数计数器

        self.random_state = np.array([])    #初始化随机状态为空

        self.state_now = np.array([])       #初始化当前状态为空

        self.state_next = np.array([])       #初始化下一刻状态为空

        self.finish = False                  #初始化完成状态

    def reset(self):

        self.cnt_step = 0 #初始化步数计数器

        self.random_state = np.array([])    #初始化随机状态为空

        self.state_now = np.array([])       #初始化当前状态为空

        self.state_next = np.array([])       #初始化下一刻状态为空

        self.finish = False                  #初始化完成状态
        #初始化应该清空所有参数


        self.state_low = np.array([ 2.,6.0,9.5,0.2,0.2,0.2,0.2])
        self.state_high = np.array([5.,9.0,12.0,0.8,0.8,0.8,0.8])

        for i in range(self.state_low.shape[0]):
            num = random.uniform(self.state_low[i],self.state_high[i])
            num = float('{:.4f}'.format(num) )
            self.random_state = np.append(self.random_state,num)

        self.cnt_step += 1

        return self.random_state

    def step(self,action):

        if self.state_now.shape[0] == 0:#如果当前状态为空，则为随机初始的第一步，将随机初始化的状态赋值给当前态

            self.state_now = self.random_state

            self.state_next = self.state_now + action

        else:#如果不为空，则说明不是第一步，则在此基础上加一个动作转移至下一个状态

            self.state_now = self.state_next #上一时刻的状态变为当前状态，当前状态加上action变为下一时刻的状态

            self.state_next = self.state_now + action


        total_data = np.vstack((agent.x_train_ad,self.state_next))#把下一时刻的预测值加入
        total_data = np.vstack((total_data,self.state_now))##再加入当前状态

        total_data = agent.scaler1.fit_transform(total_data)#归一化

        res_now = self.model(total_data)[-1,:]

        res_next = self.model(total_data)[-2,:]

        reward = 100 * ( res_next - res_now ) #奖励,扩大一百倍看看


        # file_pointer = open('RL_optimal_result/test_result.dat','a',encoding='utf-8')
        # file_pointer.write('state:'+str(self.state_low)+','+'next state:'+str(self.state_next)+',' + 'Reward:'+str( reward.numpy()[0]) )
        # file_pointer.write('\n')
        #
        # file_pointer.close()



        for i in range(self.state_low.shape[0]):

            if self.state_next[i] < self.state_low[i] or self.state_next[i] > self.state_high[i]:
                self.finish = True
            else:
                self.finish = False

        #返回下时刻的状态（32），奖励，是否完成，以及相关信息
        self.cnt_step += 1
        return ( self.state_next.astype(np.float32),reward.numpy()[0],self.finish,{'hello'},res_next.numpy()[0] )
