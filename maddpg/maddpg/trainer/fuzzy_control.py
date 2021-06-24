import numpy as np
import random
import maddpg.util.MDS as MDS
import copy
import tensorflow as tf
import maddpg.common.tf_util as U

class FuzzyTrainer():
  def __init__(self, name, agent_index, args):
    self.name = name
    self.agent_index = agent_index
    self.args = args
    self.dt = 0.1
    self.num_agents = 4
    self.ideal_side_len = args.ideal_side_len  # 弹簧原长
    self.ideal_topo_point = [[], []]
    self.v_max = 1.0
    # for i in range(self.num_agents):
    #   self.ideal_topo_point[0].append(self.ideal_side_len / np.sqrt(2) * np.cos(i / self.num_agents * 2 * np.pi)/100)
    #   self.ideal_topo_point[1].append(self.ideal_side_len / np.sqrt(2) * np.sin(i / self.num_agents * 2 * np.pi)/100)
    self.ideal_topo_point = [[0, 4, 4, 0], [0, 0, 4, 4]]
    self.ideal_topo_point = np.array(self.ideal_topo_point)

  def action(self, obs):
    obs_dis_n = obs[:self.num_agents - 1]
    obs_ang_n = obs[self.num_agents - 1:2 * (self.num_agents - 1)]
    real_x = obs[2 * (self.num_agents - 1):2 * (self.num_agents - 1) + self.num_agents]
    real_y = obs[2 * (self.num_agents - 1) + self.num_agents:2 * (self.num_agents - 1) + 2 * self.num_agents]
    obs_yaw = obs[2 * (self.num_agents - 1) + 2 * self.num_agents:2 * (
              self.num_agents - 1) + 2 * self.num_agents + self.num_agents]

    # acc_val_n = [self.k * (self.ideal_len - obs_dis) for obs_dis in obs_dis_n]
    obs_d = np.zeros((self.num_agents, self.num_agents))
    obs_theta = np.zeros((self.num_agents, self.num_agents))
    for i in range(self.num_agents):
      for j in range(i + 1):
        if i - j:
          obs_d[i, j] = np.linalg.norm([real_y[j] - real_y[i], real_x[j] - real_x[i]]) ** 2
          obs_d[j, i] = np.linalg.norm([real_y[j] - real_y[i], real_x[j] - real_x[i]]) ** 2
          obs_theta[i, j] = np.arctan2(real_y[j] - real_y[i], real_x[j] - real_x[i]) - obs_yaw[i]
          obs_theta[j, i] = np.arctan2(real_y[i] - real_y[j], real_x[i] - real_x[j]) - obs_yaw[j]

    X_est = MDS.MDS_loc(np.copy(obs_d), np.copy(obs_theta), self.num_agents)

    X_real = np.zeros(X_est.shape)

    for i in range(len(real_x)):
      X_real[0][i] = copy.deepcopy(real_x[i])
      X_real[1][i] = copy.deepcopy(real_y[i])

    _, X_final = MDS.MDS_relative_error(copy.deepcopy(X_real), copy.deepcopy(X_est), self.num_agents)
    _, X_ideal2 = MDS.MDS_relative_error(copy.deepcopy(X_final), copy.deepcopy(self.ideal_topo_point), self.num_agents)
    error, _ = MDS.MDS_relative_error(copy.deepcopy(X_real), copy.deepcopy(self.ideal_topo_point), self.num_agents)

    # fuzzy control
    vset = np.clip(2 * np.exp(-1/np.linalg.norm(X_final[:,self.agent_index]-X_ideal2[:,self.agent_index])), -self.v_max, self.v_max)
    angle_des = np.arctan2(X_ideal2[1,self.agent_index]-X_final[1,self.agent_index], X_ideal2[0,self.agent_index]-X_final[0,self.agent_index]) - obs_yaw[self.agent_index]
    angle_des = self.Norm_angle(angle_des)

    turn_head = 0
    if angle_des > 0.75 * np.pi and angle_des < 1.25*np.pi:
      angle_des -= np.pi
      turn_head = np.pi
    wset = angle_des/2/np.pi * 2

    return [wset, turn_head, vset, 0.0]

  def Norm_angle(self, ang):
    if ang > 2* np.pi:
      ang -= 2* np.pi
    elif ang < -2 * np.pi:
      ang += 2 * np.pi
    return ang
