import tensorlayer as tl
from tensorlayer.layers import Dense
from tensorlayer.models import Model
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

class PolicyNetwork(Model):
    ''' the network for generating non-determinstic (Gaussian distributed) action from the state input '''

    def __init__(self, num_inputs, num_actions, hidden_dim, action_range=1., init_w=3e-3):
        super(PolicyNetwork, self).__init__()
        # w_init = tf.keras.initializers.glorot_normal(seed=None)
        w_init = tf.random_uniform_initializer(-init_w, init_w)
        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=num_inputs, name='policy1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy2')
        self.linear3 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy3')
        self.output_linear = Dense(n_units=num_actions, W_init=w_init, \
        b_init=tf.random_uniform_initializer(-init_w, init_w), in_channels=hidden_dim, name='policy_output') #输出一个动作，维度是num_actions
        self.action_range = action_range    #动作范围
        self.num_actions = num_actions      #动作的维度
        self.tfd = tfp.distributions
        self.Normal = self.tfd.Normal  # 默认的高斯噪声

    def forward(self, state):
        x = self.linear1(state)
        x = self.linear2(x)
        x = self.linear3(x)
        output = tf.nn.tanh(self.output_linear(x))  # unit range output [-1, 1]
        return output   #归一化到-1，1之间了

    def evaluate(self, state, eval_noise_scale):    #这个预估下一时刻Q值的函数
        state = state.astype(np.float32)        #状态的type整理
        action = self.forward(state)            #通过state计算action，注意这里action范围是[-1,1]
        action = self.action_range * action     #映射到游戏的action取值范围
        # add noise
        normal = self.Normal(0, 1)                   #建立一个正态分布
        eval_noise_clip = 2 * eval_noise_scale  #对噪声进行上下限裁剪。噪声取多大合适呢，有待考证，暂时取动作的2倍，正负0.06之间
        noise = normal.sample(action.shape) * eval_noise_scale      #弄个一个noisy和action的shape一致，然后乘以scale
        noise = tf.clip_by_value(noise, -eval_noise_clip, eval_noise_clip)  #对noisy进行剪切，不要太大也不要太小
        action = action + noise                 #action加上噪音，噪音和动作的比例多少合适呢
        return action #返回有噪声的动作

    #输入state，输出action
    def get_action(self, state, explore_noise_scale):
        ''' generate action with state for interaction with envronment '''  #实际与环境的交互
        action = self.forward([state])          #这里的forward函数，就是输入state，然后通过state输出action。只不过形式不一样而已。最后的激活函数式tanh，所以范围是[-1, 1]
        action = action.numpy()[0] * self.action_range              #获得的action变成矩阵。
        # add noise
        normal = self.Normal(0, 1)                   #生成normal这样一个正态分布
        noise = normal.sample(action.shape) * explore_noise_scale       #在正态分布中抽样一个和action一样shape的数据，然后乘以scale
        action = action + noise     #action乘以动作的范围，加上noise
        return action.numpy()#一个收集样本的噪声和一个与环境交换的噪声

    def sample_action(self, ):#随机取一个动作
        ''' generate random actions for exploration '''
        a = tf.random.uniform([self.num_actions], -1, 1)
        return self.action_range * a.numpy()   #随机抽取一个动作
