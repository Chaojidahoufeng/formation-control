import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U

class SpringTrainer():
  def __init__(self, name, agent_index, args):
    self.name = name
    self.agent_index = agent_index
    self.args = args
    self.k = 0.5 # 劲度系数
    self.ideal_len = args.ideal_side_len # 弹簧原长

  def action(self, obs):
    obs_dis_n = obs[:len(obs)//2]
    obs_ang_n = obs[len(obs)//2:]

    obs_dis_n *= 100 # convert to cm

    acc_val_n = [self.k * (self.ideal_len - obs_dis) for obs_dis in obs_dis_n]
    acc_val_sum = [0,0]
    for i in range(len(acc_val_n)):
      acc_val_sum[0] += - acc_val_n[i] * np.cos(obs_ang_n[i])
      acc_val_sum[1] += - acc_val_n[i] * np.sin(obs_ang_n[i])
    
    return acc_val_sum
