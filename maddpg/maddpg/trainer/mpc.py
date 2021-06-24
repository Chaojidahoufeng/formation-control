import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U
import maddpg.util.MDS as MDS
import copy

class MPCTrainer():
  def __init__(self, name, agent_index, args):
    self.name = name
    self.agent_index = agent_index
    self.args = args
    self.n_mpc = 2 # 劲度系数
    self.dt = 0.1
    self.num_agents = 4
    self.ideal_side_len = args.ideal_side_len # 弹簧原长
    self.ideal_topo_point = [[], []]
    self.v_max = 0.5
    # for i in range(self.num_agents):
    #   self.ideal_topo_point[0].append(self.ideal_side_len / np.sqrt(2) * np.cos(i / self.num_agents * 2 * np.pi)/100)
    #   self.ideal_topo_point[1].append(self.ideal_side_len / np.sqrt(2) * np.sin(i / self.num_agents * 2 * np.pi)/100)
    self.ideal_topo_point = [[0,4,4,0],[0,0,4,4]]
    self.ideal_topo_point = np.array(self.ideal_topo_point)

  def action(self, obs):
    obs_dis_n = obs[:self.num_agents-1]
    obs_ang_n = obs[self.num_agents-1:2*(self.num_agents-1)]
    real_x = obs[2*(self.num_agents-1):2*(self.num_agents-1)+self.num_agents]
    real_y = obs[2*(self.num_agents-1)+self.num_agents:2*(self.num_agents-1)+2*self.num_agents]
    obs_yaw = obs[2*(self.num_agents-1)+2*self.num_agents:2*(self.num_agents-1)+2*self.num_agents+self.num_agents]

    #acc_val_n = [self.k * (self.ideal_len - obs_dis) for obs_dis in obs_dis_n]
    obs_d = np.zeros((self.num_agents,self.num_agents))
    obs_theta = np.zeros((self.num_agents, self.num_agents))

    for i in range(self.num_agents):
      for j in range(i+1):
        if i-j:
          obs_d[i,j] = np.linalg.norm([real_y[j]-real_y[i], real_x[j]-real_x[i]])**2
          obs_d[j,i] = np.linalg.norm([real_y[j]-real_y[i], real_x[j]-real_x[i]])**2
          obs_theta[i, j] = np.arctan2(real_y[j]-real_y[i], real_x[j]-real_x[i]) - obs_yaw[i]
          obs_theta[j, i] = np.arctan2(real_y[i]-real_y[j], real_x[i]-real_x[j]) - obs_yaw[j]

    X_est = MDS.MDS_loc(np.copy(obs_d), np.copy(obs_theta), self.num_agents)

    X_real = np.zeros(X_est.shape)

    for i in range(len(real_x)):
      X_real[0][i] = copy.deepcopy(real_x[i])
      X_real[1][i] = copy.deepcopy(real_y[i])

    _, X_final = MDS.MDS_relative_error(copy.deepcopy(X_real),copy.deepcopy(X_est), self.num_agents)
    _, X_ideal2 = MDS.MDS_relative_error(copy.deepcopy(X_final),copy.deepcopy(self.ideal_topo_point), self.num_agents)
    error, _ = MDS.MDS_relative_error(copy.deepcopy(X_real),copy.deepcopy(self.ideal_topo_point), self.num_agents)
    #print(error)
    # MPC start
    X0 = np.zeros((3,1))
    X0[0:2] = X_final[0:2, self.agent_index].reshape(2,1)
    X0[2] = obs_yaw[self.agent_index]
    X_s = np.zeros((3*self.n_mpc, 1))

    eigenvector = np.array([[X_ideal2[0, self.agent_index] - X_final[0, self.agent_index]],[X_ideal2[1, self.agent_index] - X_final[1, self.agent_index]]])
    eigenvector = eigenvector / np.linalg.norm(eigenvector)

    X_s[0:3] = X0 + np.concatenate((self.v_max*self.dt*eigenvector, np.arctan2(eigenvector[1], eigenvector[0]).reshape(1,1)))
    for s in range(2, self.n_mpc+1):
      X_s[3*s-3:3*s] = X_s[3*s-6:3*s-3] + np.concatenate((self.v_max*self.dt*eigenvector, np.zeros((1,1))))

    # X0_t = np.array([[1],[2],[0]])
    # X_S1 = np.array([[1],[4],[1],[1],[5],[2]])
    # X_S2 = np.array([[2], [-2], [0], [3], [-3], [1]])
    # u1 = self.U_MPC(X0_t,X_S1, self.n_mpc,self.dt)
    # u2 = self.U_MPC(X0_t, X_S2, self.n_mpc, self.dt)
    # print(u1,u2)
    u_mpc = self.U_MPC(X0, X_s, self.n_mpc, self.dt)
    # X_real = np.array([[-2.15910391154640,2.70963826476583,-3.76488351429127,-3.24226901349044], [1.6433, 1.9486, 2.3672, 2.7094]])
    # X_est = np.array([[0.496944422219022,-4.314933957635737,2.068201525201507,1.749788010215208],[0.591045129482105,-0.140798282633142,-0.356188828975279,-0.094058017873685]])
    # print(MDS.MDS_relative_error(X_real,X_est,4))
    
    return u_mpc

  def U_MPC(self, X0, X_s, n_mpc, dt):
    zero_threshold = 1e-3

    phi = np.squeeze(X0[2])
    #print(phi)
    a = np.array([[np.cos(phi)],[np.sin(phi)]])

    b = X_s[0:2] - X0[0:2]
    U = [0.0, 0.0, 0.0, 0.0]
    if np.linalg.norm(b) > zero_threshold:
      theta = np.squeeze(np.arccos(np.matmul(a.T,b)/np.linalg.norm(a)/np.linalg.norm(b)))
      RM = np.array([[np.cos(-phi), -np.sin(-phi)],[np.sin(-phi), np.cos(-phi)]])
      btemp = np.matmul(RM, b)
      if btemp[1]<0:
        theta = -theta

      U[0] = 2 * theta / dt

      speed = 0.5*np.linalg.norm(b)/np.sin(theta)
      U[2] = np.clip(U[0] *speed, -self.v_max, self.v_max)


    # for s in range(2, n_mpc+1):
    #   phi = phi + (U[4*(s-1)-4] - U[4*(s-1)-3]) * dt
    #   a = np.array([[np.cos(phi)], [np.sin(phi)]])
    #   b = X_s[3*s-3:3*s-1] - X_s[3*s-6:3*s-4]
    #
    #   U = [0.0, 0.0, 0.0, 0.0]
    #   if np.linalg.norm(b) > zero_threshold:
    #     theta = np.squeeze(np.arccos(np.matmul(a.T, b) / np.linalg.norm(a) / np.linalg.norm(b)))
    #     RM = np.array([[np.cos(-phi), -np.sin(-phi)], [np.sin(-phi), np.cos(-phi)]])
    #     btemp = np.matmul(RM, b)
    #
    #     if btemp[1] < 0:
    #       theta = -theta
    #
    #     if theta > 0:
    #       U[0] = 2 * theta / dt
    #     else:
    #       U[1] = abs(2 * theta / dt)
    #
    #     speed = 0.5 * np.linalg.norm(b) / np.sin(theta)
    #
    #     U[2] = (U[0] - U[1]) * speed

    return U

  def wrap2pi(self, ang):
    if abs(ang) > np.pi:
      ang -= np.sign(ang) * 2 * np.pi
    return ang