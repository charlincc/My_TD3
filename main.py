from ReplayBuffer_file import ReplayBuffer
from ENV import MY_ENV
from TD3_Trainer_file import TD3_Trainer
import argparse
import random
import time
import numpy as np



if __name__ == '__main__':
    # add arguments in command  --train/test
    parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
    parser.add_argument('--open', dest='open', action='store_true', default=True)
    parser.add_argument('--close', dest='close', action='store_true', default=False)
    args = parser.parse_args()
    #####################  hyper parameters  ####################
    # choose env
    action_range = 0.03  # scale action, [-action_range, action_range]
    # RL training
    max_train_eposide = 2000  # 训练的最大的探索总EPOCHE
    max_test_eposide = 5  # 测试的最大的探索总EPOCHE
    test_frames = 300  # 测试的最大的探索总步数
    max_train_steps = 100  # train每个eposide的探索步数
    max_test_steps = 100  # test每个episode的探索步骤
    batch_size = 64  # udpate batchsize
    explore_steps = 500  # 500 for random action sampling in the beginning of training
    update_itr = 3  # 软更新的间隔，每几步更新一次
    hidden_dim = 128  # 每个隐藏层的神经元数
    q_lr = 3e-4  # critic的学习率
    policy_lr = 3e-4  # actor的学习率
    policy_target_update_interval = 3  # delayed steps for updating the policy network and target networks
    explore_noise_scale = 0.03  # range of action noise for exploration
    eval_noise_scale = 0.03  # range of action noise for evaluation of action value
    reward_scale = 10.  # value range of reward
    replay_buffer_size = 5e5  # size of replay buffer
    env = MY_ENV()      #环境
    # initialization of buffer
    replay_buffer = ReplayBuffer(replay_buffer_size)#初始化记忆回放池的大小
    # initialization of trainer
    td3_trainer=TD3_Trainer(replay_buffer, hidden_dim=hidden_dim, policy_target_update_interval=policy_target_update_interval, \
    action_range=action_range, q_lr=q_lr, policy_lr=policy_lr )#初始化TD3网络的各种网络
    # set train mode
    td3_trainer.q_net1.train()
    td3_trainer.q_net2.train()
    td3_trainer.target_q_net1.train()
    td3_trainer.target_q_net2.train()
    td3_trainer.policy_net.train()
    td3_trainer.target_policy_net.train()#实例化对象
    # training loop
    if args.open:
        #清空以前的数据
        print('开始强化训练！！！')#
        file_pointer = open('RL_train_data/train_data.dat','w',encoding='utf-8')
        file_pointer.close()
        total_step = 0                                   #累计总步数
        train_eposide = 0                               #累计eposide
        rewards = []                                    #记录每个EP的总reward
        t0 = time.time()
        while train_eposide < max_train_eposide:           #小于最大eposode，就继续训练
            state = env.reset()                 #初始化state
            state = state.astype(np.float32)    #整理state的类型
            if total_step < 1:                   #第一次的时候，要进行初始化trainer
                print('intialize')
                _ = td3_trainer.policy_net([state])  # need an extra call here to make inside functions be able to use model.forward
                _ = td3_trainer.target_policy_net([state])
            # 开始训练
            cnt = 0                 #算一回合运行的步数
            episode_reward = 0      #算一回合的累计奖励
            for step in range(max_train_steps):

                if total_step > explore_steps:       #如果小于500步，就随机，如果大于就用get-action
                    action = td3_trainer.policy_net.get_action(state, explore_noise_scale=0.01)  #带有noisy的action
                else:
                    action = td3_trainer.policy_net.sample_action()##########
                # 与环境进行交互
                next_state, reward, done, n_nn,next_result = env.step(action)
                next_state = next_state.astype(np.float32)
                done = 1 if done ==True else 0#三目操作，真返回第一个，加返回第三个
                #记录数据在replay_buffer
                replay_buffer.push(state, action, reward, next_state, done)
                #赋值state，累计总reward，步数
                episode_reward += reward
                state = next_state
                total_step += 1#累计步数
                cnt += 1
                #如果数据超过一个batch_size的大小，那么就开始更新
                if len(replay_buffer) > batch_size:
                    for i in range(update_itr):         #注意：这里更新可以更新多次！
                        td3_trainer.update(batch_size, eval_noise_scale=0.05, reward_scale=1.)
                #把训练数据写进去文件里面去
                file_pointer = open('RL_train_data/train_data.dat','a',encoding='utf-8')
                file_pointer.write('state:')
                for i in next_state:
                    file_pointer.write(str(i)+',')
                file_pointer.write('\n')
                file_pointer.write('L/D coe:'+str(next_result))
                file_pointer.write('\n')
                file_pointer.close()
                if done:#done为则结束
                    train_eposide+=1
                    break#循环条件出错，有待更正
            if cnt == max_train_steps:
                train_eposide+=1
            print('Episode: {}  | Episode Reward: {:.4f}  | total step: {}  | Running Time: {:.4f}'\
            .format(train_eposide, episode_reward,cnt,time.time()-t0))
            rewards.append(episode_reward)
            #写一个eposide的总奖励
            file_pointer = open('RL_train_data/train_data.dat','a',encoding='utf-8')
            file_pointer.write('Eposide reward:' + str(episode_reward))
            file_pointer.write('\n')
            file_pointer.write('\n')
            file_pointer.close()
        td3_trainer.save_weights()
    if args.open:#train和test互换，实现测试和训练的互换
        #清空以前的数据
        print('开始强化测试！！！')
        file_pointer = open('RL_optimal_result/test_result.dat','w',encoding='utf-8')
        file_pointer.close()
        file_pointer = open('RL_optimal_result/L_D_coe.dat','w',encoding='utf-8')
        file_pointer.close()
        #定义计数器和容器
        frame_idx = 0
        rewards = []
        t0 = time.time()
        test_eposide = 0
        td3_trainer.load_weights()
        while test_eposide < 50:
            state = env.reset()
            state = state.astype(np.float32)
            episode_reward = 0
            cnt = 0
            if frame_idx < 1:
                print('intialize')
                _ = td3_trainer.policy_net(
                    [state]
                )  # need an extra call to make inside functions be able to use forward
                _ = td3_trainer.target_policy_net([state])
            for step in range(200):
                action = td3_trainer.policy_net.get_action(state, explore_noise_scale=0.1)
                next_state, reward, done, _,next_result = env.step(action)
                file_pointer = open('RL_optimal_result/L_D_coe.dat','a',encoding='utf-8')
                file_pointer.write(str(next_result))
                file_pointer.write('\n')
                file_pointer.close()
                #把相关的状态信息写到文件里面
                file_pointer = open('RL_optimal_result/test_result.dat', 'a', encoding='utf-8')
                for i in next_state:
                    file_pointer.write(str(i)+',')
                file_pointer.write('\n')
                file_pointer.write('L/D:' + str(next_result))
                file_pointer.write('\n')
                file_pointer.close()
                next_state = next_state.astype(np.float32)
                done = 1 if done == True else 0
                state = next_state
                episode_reward += reward
                frame_idx += 1
                cnt+=1
                if done:
                    test_eposide+=1
                    break
            file_pointer = open('RL_optimal_result/test_result.dat','a',encoding='utf-8')
            file_pointer.write('total reward:' + str(episode_reward) )
            file_pointer.write('\n')
            file_pointer.write('\n')
            file_pointer.close()
            #写一个空行，分割开
            file_pointer = open('RL_optimal_result/L_D_coe.dat', 'a', encoding='utf-8')
            file_pointer.write('\n')
            file_pointer.close()
            print('Episode: {}  | Episode Reward: {:.4f}  | total step: {}| Running Time: {:.4f}'\
            .format(test_eposide, episode_reward,cnt ,time.time()-t0 ) )
            rewards.append(episode_reward)
            print('test')

            