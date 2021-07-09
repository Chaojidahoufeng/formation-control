import gym
import tensorflow as tf
import collections
import random
from tensorflow.keras import layers,models,optimizers
import numpy as np
# 算法输入：迭代轮数T，状态特征维度n, 动作集A, 步长α，衰减因子γ, 探索率ϵ, 当前Q网络Q，目标Q网络Q′, 批量梯度下降的样本数m,目标Q网络参数更新频率C。
#
# 　　　　输出：Q网络参数
#
# 　　　　1. 随机初始化所有的状态和动作对应的价值Q.  随机初始化当前Q网络的所有参数w,初始化目标Q网络Q′的参数w′=w。清空经验回放的集合D。
#
# 　　　　2. for i from 1 to T，进行迭代。
#
# 　　　　　　a) 初始化S为当前状态序列的第一个状态, 拿到其特征向量ϕ(S)
# 　　　　　　b) 在Q网络中使用ϕ(S)作为输入，得到Q网络的所有动作对应的Q值输出。用ϵ−贪婪法在当前Q值输出中选择对应的动作A
# 　　　　　　c) 在状态S执行当前动作A,得到新状态S′对应的特征向量ϕ(S′)和奖励R$,是否终止状态is_end
#
# 　　　　　　d) 将{ϕ(S),A,R,ϕ(S′),is_end}这个五元组存入经验回放集合D
# 　　　　　　e) S=S′
# 　　　　　　f)  从经验回放集合D中采样m个样本{ϕ(Sj),Aj,Rj,ϕ(S′j),is_endj},j=1,2.,,,m，计算当前目标Q值yj：
# yj={RjRj+γmaxa′Q′(ϕ(S′j),A′j,w′)is_endjistrueis_endjisfalse
# 　　　　　　g)  使用均方差损失函数1m∑j=1m(yj−Q(ϕ(Sj),Aj,w))2，通过神经网络的梯度反向传播来更新Q网络的所有参数w
# 　　　　　　h) 如果i%C=1,则更新目标Q网络参数w′=w
# 　　　　　　i) 如果S′是终止状态，当前轮迭代完毕，否则转到步骤b)

class dqn:
    def __init__(self,state_dim,action_dim):
            self.s_dim=state_dim
            self.a_dim=action_dim
            self.memsize = 2000
            self.update_freq = 200
            self.model = self.create_model()
            self.tmodel = self.create_model()
            self.memory_counter = 0  # 记忆库记数
            self.memory = np.zeros((self.memsize, state_dim * 2 + 2))  # 初始化记忆库
            self.step=0





    def create_model(self):
        model = models.Sequential()
        model.add(layers.Input(shape=(self.s_dim)))
        model.add(layers.Dense(60, activation='relu', name='l1'))
        model.add(layers.Dense(100, activation='relu', name='l2'))
        model.add(layers.Dense(self.a_dim, name='l3'))
        model.compile(loss='mean_squared_error',
                      optimizer=optimizers.Adam(0.001))
        return model

    def choose_act(self,s,epsilon=0.2):
        if np.random.uniform() < epsilon-self.step * 0.0002:#random property decrease
            return np.random.choice(self.a_dim)
        return np.argmax(self.model.predict(np.array([s]))[0])

    def store(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % self.memsize
        self.memory[index, :] = transition
        self.memory_counter += 1

    def train(self, batch_size=64, lr=1, factor=0.95):
        if self.memory_counter < self.memsize:
            return
        self.step += 1
        # 每 update_freq 步，将 model 的权重赋值给 target_model
        if self.step % self.update_freq == 0:
            self.tmodel.set_weights(self.model.get_weights())

            # 抽取记忆库中的批数据
        sample_index = np.random.choice(self.memsize,batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = np.array(b_memory[:, :self.s_dim])
        b_a = np.array(b_memory[:, self.s_dim:self.s_dim + 1].astype(int))
        b_r = np.array(b_memory[:, self.s_dim + 1:self.s_dim + 2])
        b_s_ = np.array(b_memory[:, -self.s_dim:])


        Q = self.model.predict(b_s)
        Q_next = self.tmodel.predict(b_s_)

        for i in range(batch_size):
            a=b_a[i]
            reward=b_r[i]
            Q[i][a] = (1 - lr) * Q[i][a] + lr * (reward + factor * np.amax(Q_next[i]))



        self.model.fit(b_s, Q, verbose=0)

env = gym.make('MountainCar-v0')
# print(env.action_space.n)
episodes = 1000  # 训练1000次
score_list = []  # 记录所有分数
agent =dqn(env.observation_space.shape[0],env.action_space.n)
for i in range(episodes):
    s = env.reset()
    score = 0
    while True:
        env.render()
        a = agent.choose_act(s)
        s_, r, done, info = env.step(a)
        agent.store(s, a, r, s_)
        if agent.memory_counter > agent.memsize:
            agent.train()
            score += r
        s = s_
        if done:
            score_list.append(score)
            print('episode:', i, 'score:', score)
            break

env.close()





