from QNETwork_file import QNetwork
from PolicyNetwork_file import PolicyNetwork
import tensorflow as tf
import tensorlayer as tl
import numpy as np

class TD3_Trainer():

    def __init__(
            self, replay_buffer, hidden_dim, action_range, policy_target_update_interval=1, q_lr=3e-4, policy_lr=3e-4,state_dim = 7,action_dim = 7):#默认状态和动作空间为7维
        self.replay_buffer = replay_buffer
        # initialize all networks
        # 用两个Qnet来估算，doubleDQN的想法。同时也有两个对应的target_q_net
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
        self.q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q_net2 = QNetwork(state_dim, action_dim, hidden_dim)

        #一个策略网络带价值网络，解决其高估问题
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
        self.target_q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net2 = QNetwork(state_dim, action_dim, hidden_dim)


        print('Q Network (1,2): ', self.q_net1)
        print('Policy Network: ', self.policy_net)

        # initialize weights of target networks
        # 把net 赋值给target_network
        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)
        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)


        self.update_cnt = 0     #更新次数
        self.policy_target_update_interval = policy_target_update_interval      #策略网络更新频率

        self.q_optimizer1 = tf.optimizers.Adam(q_lr)
        self.q_optimizer2 = tf.optimizers.Adam(q_lr)
        self.policy_optimizer = tf.optimizers.Adam(policy_lr)

    #在网络初始化的时候进行硬更新
    def target_ini(self, net, target_net):
        ''' hard-copy update for initializing target networks '''
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(param)

        return target_net

    #在更新的时候进行软更新
    def target_soft_update(self, net, target_net, soft_tau):
        ''' soft update the target net with Polyak averaging '''
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(  # copy weight value into target parameters
                target_param * (1.0 - soft_tau) + param * soft_tau
                # 原来参数占比 + 目前参数占比
            )
        return target_net

    def update(self, batch_size, eval_noise_scale, reward_scale=10., gamma=0.9, soft_tau=1e-2):
        ''' update all networks in TD3 '''
        self.update_cnt += 1        #计算更新次数
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)     #从buffer sample数据,随机采样的，拿出来的是batch_size行的矩阵
        reward = reward[:, np.newaxis]  # expand dim， 调整形状，方便输入网络,
        done = done[:, np.newaxis]#把数组扩展成矩阵
        # 输入s',从target_policy_net计算a'。注意这里有加noisy的
        new_next_action = self.target_policy_net.evaluate(
            next_state, eval_noise_scale=eval_noise_scale
        )  # 预估下一步动作，带高斯噪声的动作
        # 归一化reward.(有正有负)
        reward = reward_scale * (reward - np.mean(reward, axis=0)) / (  #np.mean计算每一列的均值
            np.std(reward, axis=0) + 1e-6                               #计算每一列的标准差，但是为啥要对reward做归一化呢
        )  # normalize with batch mean and std; plus a small number to prevent numerical problem
        # Training Q Function
        # 把s'和a'堆叠在一起，一起输入到target_q_net。
        # 有两个qnet，我们取最小值
        target_q_input = tf.concat([next_state, new_next_action], 1)  # the dim 0 is number of samples
        target_q_min = tf.minimum(self.target_q_net1(target_q_input), self.target_q_net2(target_q_input))#状态t+1时刻的预估Q值，取较小值
        #计算target_q的值，用于更新q_net
        #之前有把done从布尔变量改为int，就是为了这里能够直接计算。
        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward，这是TD目标
        q_input = tf.concat([state, action], 1)  # input of q_net，状态s，和动作a的结合体
        #更新q_net1
        #这里其实和DQN是一样的
        with tf.GradientTape() as q1_tape:
            predicted_q_value1 = self.q_net1(q_input)   #一次前向传播，输出q值
            q_value_loss1 = tf.reduce_mean(tf.square(predicted_q_value1 - target_q_value))
            print('q1_net loss',q_value_loss1)
        q1_grad = q1_tape.gradient(q_value_loss1, self.q_net1.trainable_weights)
        self.q_optimizer1.apply_gradients(zip(q1_grad, self.q_net1.trainable_weights))
        #更新q_net2
        with tf.GradientTape() as q2_tape:
            predicted_q_value2 = self.q_net2(q_input)
            q_value_loss2 = tf.reduce_mean(tf.square(predicted_q_value2 - target_q_value))
            print('q2_net loss:',q_value_loss2)
        q2_grad = q2_tape.gradient(q_value_loss2, self.q_net2.trainable_weights)
        self.q_optimizer2.apply_gradients(zip(q2_grad, self.q_net2.trainable_weights))
        # Training Policy Function
        # policy不是经常updata的，而qnet更新一定次数，才updata一次
        if self.update_cnt % self.policy_target_update_interval == 0:   #每隔interval更新一次
            #更新policy_net
            with tf.GradientTape() as p_tape:
                # 计算 action = Policy(s)，注意这里是没有noise的
                new_action = self.policy_net.evaluate(
                    state, eval_noise_scale=0.0
                )  # no noise, deterministic policy gradients
                #叠加state和action
                new_q_input = tf.concat([state, new_action], 1)
                # ''' implementation 1 '''
                # predicted_new_q_value = tf.minimum(self.q_net1(new_q_input),self.q_net2(new_q_input))
                ''' implementation 2 '''
                predicted_new_q_value = self.q_net1(new_q_input)
                policy_loss = -tf.reduce_mean(predicted_new_q_value)    #梯度上升
                print('policy loss',policy_loss)
            p_grad = p_tape.gradient(policy_loss, self.policy_net.trainable_weights)
            self.policy_optimizer.apply_gradients(zip(p_grad, self.policy_net.trainable_weights))
            # Soft update the target nets
            # 软更新target_network三个
            self.target_q_net1 = self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
            self.target_q_net2 = self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)
            self.target_policy_net = self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

    def save_weights(self):  # save trained weights
        tl.files.save_npz(self.q_net1.trainable_weights, name='RL_model/model_q_net1.npz')
        tl.files.save_npz(self.q_net2.trainable_weights, name='RL_model/model_q_net2.npz')
        tl.files.save_npz(self.target_q_net1.trainable_weights, name='RL_model/model_target_q_net1.npz')
        tl.files.save_npz(self.target_q_net2.trainable_weights, name='RL_model/model_target_q_net2.npz')
        tl.files.save_npz(self.policy_net.trainable_weights, name='RL_model/model_policy_net.npz')
        tl.files.save_npz(self.target_policy_net.trainable_weights, name='RL_model/model_target_policy_net.npz')

    def load_weights(self):  # load trained weights
        tl.files.load_and_assign_npz(name='RL_model/model_q_net1.npz', network=self.q_net1)
        tl.files.load_and_assign_npz(name='RL_model/model_q_net2.npz', network=self.q_net2)
        tl.files.load_and_assign_npz(name='RL_model/model_target_q_net1.npz', network=self.target_q_net1)
        tl.files.load_and_assign_npz(name='RL_model/model_target_q_net2.npz', network=self.target_q_net2)
        tl.files.load_and_assign_npz(name='RL_model/model_policy_net.npz', network=self.policy_net)
        tl.files.load_and_assign_npz(name='RL_model/model_target_policy_net.npz', network=self.target_policy_net)
