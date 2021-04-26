import numpy as np
import math as math
import random as random
from numpy.linalg import norm
from multiagent.core import World, Agent, Landmark, Static_obs, Dynamic_obs
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        #rendering unit is in centimeter
        world = World()
        # set any world properties first
        world.dim_c = 0
        #world.damping = 1

        num_follower_agents = 4
        num_leader_agent = 1
        num_agents = num_leader_agent + num_follower_agents
        num_static_obs = 5
        num_landmarks = 1# tracking center and side barrier indicators
        num_agent_ray = 10
        # add agents
        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            # check if agent already collided
            agent.collide = False
            agent.leader = True if i == 0 else False
            agent.silent = False #if i > 0 else False
            #agent.adversary = True if i < num_adversaries else False
            agent.boundary = False
            agent.size = 10*np.sqrt(2)
            agent.accel = 3
            agent.max_speed = 100
            agent.u_noise = 0.2
            agent.color = np.array([0., 1, 0.45]) if not agent.leader else np.array([1.0, 1.0, 0.0])
            agent.crash = 0  # calculate how many time the agent crashed
            agent.ray = np.zeros((num_agent_ray, 2))  # num_agent_ray*[dis, ang, type]
            agent.ang_range = {'L1': np.pi/2 - 2 * np.pi / 8 - 2 * np.pi / 10, 'L2': np.pi/2 - 2 * np.pi / 8 - 1 * np.pi / 10, 'L3': np.pi/2 - 2 * np.pi / 8, 'L4': np.pi/2 - 1 * np.pi / 8, 'L5': np.pi/2,
                         'R1': -np.pi/2 + 2 * np.pi / 8 + 2 * np.pi / 10, 'R2': -np.pi/2 + 2 * np.pi / 8 + 1 * np.pi / 10, 'R3': -np.pi/2 + 2 * np.pi / 8, 'R4': -np.pi/2 + 1 * np.pi / 8, 'R5': -np.pi/2}
            # for stream function based avoidance
            agent.avoiding = [False, False]
            agent.U = [0, 0]
            agent.first_ray = [0, 0]
            agent.C = [0, 0]
            agent.C_bound = [0, 0]

        world.static_obs = [Static_obs() for _ in range(num_static_obs)]
        for i, static_obs in enumerate(world.static_obs):
            static_obs.name = 'static_obs %d' % i
            static_obs.collide = True
            static_obs.movable = False
            static_obs.boundary = False
            static_obs.size = 20*np.sqrt(2)
            static_obs.color = np.array([0.8, 0.8, 0.8])
            static_obs.state.p_vel = np.zeros(world.dim_p)
            static_obs.state.p_ang = 0.0
        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.boundary = False

            if i==0:
                landmark.center = True
                landmark.size = 5
                landmark.color = np.array([1, 1, 0.4])
                landmark.distance = 0
            else:
                landmark.center = False
                landmark.size = 20
                landmark.color = np.array([0.5, 0.5, 0.5])

        # make initial conditions
        self.reset_world(world)
        return world

    def set_path(self, path_type, start):
        if path_type == 'square':
            xs = np.array([-start, start, start, -start, -start])
            ys = np.array([-start, -start, start, start, -start])
            xys = list(zip(xs, ys))
        elif path_type == 'line':
            xs = np.array([start[0], 4])
            ys = np.array([start[1], start[1]])
            xys = list(zip(xs, ys))
        return xys

    def reset_world(self, world):
        world.time = 0
        world.distance = 0
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.crash = 0  # reset collision counter
        # random properties for landmarks
        num_dynamic_obs = 0#np.random.randint(1,3)
        world.dynamic_obs = [Dynamic_obs() for _ in range(num_dynamic_obs)]

            #s.state.p_pos = np.array([500.0, 50 * i])

        # set initial states of all agents
        agents_ctr = np.array((0.0, 0.0))
        # add Path
        path_type = 'line'
        world.start = agents_ctr
        world.goal = agents_ctr + 10000 * np.array([0, 1])
        #world.path = self.set_path(path_type, world.start)

        #world.station_num = len(world.path)
        world.station = 1
        #goal_vect = world.goal-agents_ctr
        leader_vel = np.random.uniform(0.1, 0.3)

        for e, agent in enumerate(world.agents):
            agent.agents_ctr = agent.agents_ctr_prev = agents_ctr
            agent.state.p_ang = math.atan2(world.goal[1] - world.start[1], world.goal[0] - world.start[0])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.p_omg = 0.0
            agent.state.c = np.zeros(world.dim_c)

            if agent.leader:
                agent.state.p_pos = agents_ctr
            else:
                ang = world.agents[0].state.p_ang+(e-1)*2*np.pi/(len(world.agents)-1)
                agent.state.p_pos = agent.agents_ctr + 20*agent.size*np.array([math.cos(ang), math.sin(ang)])

            if not agent.leader:
                agent.dis2leader = norm(agent.state.p_pos - agents_ctr)/100
                agent.ang2leader = self.get_relAngle(world.agents[0], agent)
                agent.dis2leader_prev = None
                agent.ang2leader_prev = None
                # initialize the desire relative position of agents in the formation
                agent.d_des = agent.dis2leader
                agent.w_des = agent.ang2leader
                agent.p_des = agent.d_des * np.array([math.cos(agent.w_des), math.sin(agent.w_des)])
                agent.p_rel = agent.dis2leader * np.array([math.cos(agent.ang2leader), math.sin(agent.ang2leader)])
                agent.his = [] # initialize leader history
                agent.P = np.eye(4) # initialize follower predict variance
            else:
                agent.dis2goal = norm(agent.state.p_pos - world.goal) / 100
                agent.ang2goal = math.atan2(world.goal[1]-agent.state.p_pos[1],
                                              world.goal[0]-agent.state.p_pos[0]) - agent.state.p_ang
                agent.dis2goal_prev = None
                agent.ang2goal_prev = None

            agent.err = np.zeros(2)
            agent.err_prev = 0.0
            #agent.dis2goal = norm(agent.state.p_pos - world.goal - agent.p_des) / 100
            #agent.ang2goal = math.atan2(agent.state.p_pos[1] - world.goal[1] - agent.p_des[1], agent.state.p_pos[0] - world.goal[0] - agent.p_des[0])
            #agent.dis2goal_prev = None
            #agent.ang2goal_prev = None
            agent.state.p_pos += np.array([2*np.random.uniform(-10, 10), 2*np.random.uniform(-10, 10)])
            agent.state.p_ang += np.random.uniform(-np.pi/6,np.pi/6)
            self.set_agent_ray(agent, obs_type='obs')

        for i, dynamic_obs in enumerate(world.dynamic_obs):
            dynamic_obs.name = 'dynamic_obs %d' % i
            dynamic_obs.collide = True
            dynamic_obs.movable = True
            dynamic_obs.boundary = False
            dynamic_obs.size = 10*np.sqrt(2)
            dynamic_obs.accel = 50
            dynamic_obs.max_speed = 100
            dynamic_obs.color = np.array([0., 0., 1])

        for s in world.static_obs:
            s.state.p_pos = np.array([np.random.uniform(-400, 400), np.random.uniform(-400, 400)])
            while norm(s.state.p_pos-agents_ctr)/100 < 2 or norm(s.state.p_pos-world.goal)/100 < 2:
                s.state.p_pos = np.array([np.random.uniform(-400, 400), np.random.uniform(-400, 400)])
        for i,landmark in enumerate(world.landmarks):
            if landmark.center:
                landmark.state.p_pos = world.goal

    def benchmark_data(self, agent, world):
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def get_relAngle(self, center, target):
        ang = math.atan2(target.state.p_pos[1] - center.state.p_pos[1], target.state.p_pos[0] - center.state.p_pos[0]) - center.state.p_ang
        if abs(ang) > np.pi:
            ang -= np.sign(ang) * 2 * np.pi
        return ang

    def wrap2pi(self, ang):
        if abs(ang) > np.pi:
            ang -= np.sign(ang) * 2 * np.pi
        return ang

    def set_agent_ray(self, agent, dis2obs=None, ang2obs=None, obs_type=None):
        ang_range = list(agent.ang_range.items())
        ray_ang = -np.pi / 2
        for i in range(len(agent.ray)):
            #ray_ang = self.wrap2pi(ray_ang + ang_range[i])
            agent.ray[i][0] = 3
            agent.ray[i][1] = self.wrap2pi(ang_range[i][1])
            #agent.ray[i][2] = 0
            if ang2obs:
                for j in range(len(ang2obs)):
                    # todo: modify the detect condition
                    if abs(self.wrap2pi(ang2obs[j] - agent.ray[i][1])) < np.pi / 10 and dis2obs[j] < agent.ray[i][0]:
                        agent.ray[i][0] = abs(dis2obs[j])

    # collision detect
    def is_collision(self, center, target):
        delta_pos = center.state.p_pos - target.state.p_pos
        if (center.state.p_pos[0] - target.state.p_pos[0]) != 0:
            ang = math.atan((center.state.p_pos[1] - target.state.p_pos[1]) / (
                        center.state.p_pos[0] - target.state.p_pos[0]))
        else:
            ang = np.pi / 2
        dist = norm(delta_pos)/100
        dist_min = (center.size + target.size)*math.cos(abs(ang)-np.pi/4)/100

        return dist <= dist_min

    # return all agents that are not adversaries
    def follower_agents(self, world):
        return [agent for agent in world.agents if not agent.leader]

    # return all adversarial agents
    def leader_agents(self, world):
        return [agent for agent in world.agents if agent.leader]

    # return all static obstacles
    def get_static_obstacles(self, world):
        return [static_obs for static_obs in world.static_obs]

    # return all dynamic obstacles
    def get_dynamic_obstacles(self, world):
        return [dynamic_obs for dynamic_obs in world.dynamic_obs]

    # return current state of formation center
    def get_formation_center(self, world):
        return world.agents[0].p_pos

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        #boundary_reward = -10 if self.outside_boundary(agent) else 0
        main_reward = self.leader_reward(agent, world) if agent.leader else self.agent_reward(agent, world)
        return main_reward

    def outside_boundary(self, entity):
        # 20190711 restrict the agents in the frame
        if entity.state.p_pos[0] + entity.size > 5:
            entity.state.p_pos[0] = 5 - entity.size
            #return True
        if entity.state.p_pos[0] - entity.size < -5:
            entity.state.p_pos[0] = -5 + entity.size
            #return True
        if entity.state.p_pos[1] + entity.size > 5:
            entity.state.p_pos[1] = 5 - entity.size
            #return True
        if entity.state.p_pos[1] - entity.size < -5:
            entity.state.p_pos[1] = -5 + entity.size
            #return True
        #return False

    def get_stream_value(self, agent):
        obstacle_r = 1.5*20*np.sqrt(2)/100
        detect_range = 3
        err = 0
        # check avoiding state
        if not agent.avoiding[0]:
            # check if obstacle detected at left side
            agent.C[0] = 0
            for n in range(0, len(agent.ray)//2):
                if agent.ray[n][0] < detect_range:
                    agent.U[0] = 1#norm(agent.state.p_vel)/100
                    agent.first_ray[0] = n

                    x = - (agent.ray[n][0] + obstacle_r) * np.cos(agent.ray[n][1])
                    y = - (agent.ray[n][0] + obstacle_r) * np.sin(agent.ray[n][1])
                    agent.C_bound[0] = - agent.U[0] * (3 * agent.size / 100 + obstacle_r) * (1 - np.square(obstacle_r) / ((3 * agent.size / 100 + obstacle_r) ** 2))
                    #x = - agent.ray[n][0] * np.cos(agent.ray[n][1])
                    #y = - agent.ray[n][0] * np.sin(agent.ray[n][1])
                    agent.C[0] = min(agent.U[0] * y * (1 - np.square(obstacle_r) / (x ** 2 + y ** 2)), agent.C_bound[0])
                    agent.avoiding[0] = True
                    break

        if not agent.avoiding[1]:
            # check if obstacle detected at right side
            agent.C[1] = 0
            for n in range(len(agent.ray)//2, len(agent.ray)):
                if agent.ray[n][0] < detect_range:
                    agent.U[1] = 1#norm(agent.state.p_vel)/100
                    agent.first_ray[1] = n - len(agent.ray)//2

                    x = - (agent.ray[n][0] + obstacle_r) * np.cos(agent.ray[n][1])
                    y = - (agent.ray[n][0] + obstacle_r) * np.sin(agent.ray[n][1])

                    agent.C_bound[1] = agent.U[1] * (3 * agent.size / 100 + obstacle_r) * (1 - np.square(obstacle_r) / ((3 * agent.size / 100 + obstacle_r) ** 2))
                    #x = - agent.ray[n][0] * np.cos(agent.ray[n][1])
                    #y = - agent.ray[n][0] * np.sin(agent.ray[n][1])
                    agent.C[1] = max(agent.U[1] * y * (1 - np.square(obstacle_r) / (x ** 2 + y ** 2)), agent.C_bound[1])
                    agent.avoiding[1] = True
                    break

        # check if left and right sensors sensed same object
        if agent.first_ray[0] == agent.first_ray[1] == 0:
            # choose the shorter side to follow the streamline
            if agent.ray[0][0] > agent.ray[len(agent.ray)//2][0]:
                agent.avoiding[0] = False
            elif agent.ray[len(agent.ray)//2][0] > agent.ray[0][0]:
                agent.avoiding[1] = False

        # calculate streamline following error
        if agent.avoiding[0]:
            for n in range(0, len(agent.ray) // 2):
                if agent.ray[n][0] < detect_range:

                    x = - (agent.ray[n][0] + obstacle_r) * np.cos(agent.ray[n][1])
                    y = - (agent.ray[n][0] + obstacle_r) * np.sin(agent.ray[n][1])

                    #x = - agent.ray[n][0] * np.cos(agent.ray[n][1])
                    #y = - agent.ray[n][0] * np.sin(agent.ray[n][1])
                    '''
                    u = norm(agent.state.p_vel)/100 * np.cos(agent.ray[n][1])
                    v = norm(agent.state.p_vel)/100 * np.sin(agent.ray[n][1])
                    u_target = agent.U[0] * [1 - np.square(obstacle_r) * (x ** 2 - y ** 2) / np.square(x ** 2 + y ** 2)]
                    v_target = -
                    '''
                    c = agent.U[0] * y * (1 - np.square(obstacle_r) / (x ** 2 + y ** 2))
                    #print("left: ", c, agent.C[0])
                    err += np.sqrt(np.square(c - agent.C[0]))

                    if n < agent.first_ray[0]:
                        # new obstacle detected
                        agent.U[0] = 1#norm(agent.state.p_vel)/100
                        agent.C[0] = min(c, agent.C_bound[0])
                    agent.first_ray[0] = n
                    break
                if n == len(agent.ray) // 2 - 1:
                    # if traverse to the last sensor and still no obstacle detected
                    agent.avoiding[0] = False

        if agent.avoiding[1]:
            for n in range(len(agent.ray)//2, len(agent.ray)):
                if agent.ray[n][0] < detect_range:

                    x = - (agent.ray[n][0] + obstacle_r) * np.cos(agent.ray[n][1])
                    y = - (agent.ray[n][0] + obstacle_r) * np.sin(agent.ray[n][1])

                    #x = - agent.ray[n][0] * np.cos(agent.ray[n][1])
                    #y = - agent.ray[n][0] * np.sin(agent.ray[n][1])
                    c = agent.U[1] * y * (1 - np.square(obstacle_r) / (x ** 2 + y ** 2))
                    #print("right: ", c, agent.C[1])
                    err += np.sqrt(np.square(c - agent.C[1]))
                    if n - len(agent.ray)//2 < agent.first_ray[1]:
                        # new obstacle detected
                        agent.U[1] = 1#norm(agent.state.p_vel)/100
                        agent.C[1] = max(c, agent.C_bound[1])
                    agent.first_ray[1] = n - len(agent.ray)//2
                    break
                if n == len(agent.ray) - 1:
                    # if traverse to the last sensor and still no obstacle detected
                    agent.avoiding[1] = False
        #print("error: ", err)
        return err

    def agent_reward(self, agent, world):
        # 20190711
        # Agents are rewarded based on
        # 1.whether any collision happened
        # 2.the distance between formation center and desire path
        # 3.how the shape of formation is maintained

        # step reward
        #rew = -0.3
        rew = 0

        # parameters for collision reward
        alpha = 1
        beta = 3
        # get the angle of target position
        target_ang = self.wrap2pi(np.arctan2(agent.err[1], agent.err[0]))
        leader_ang = self.get_relAngle(agent, world.agents[0])
        agents = self.follower_agents(world) + self.leader_agents(world)
        static_obs = self.get_static_obstacles(world)
        dynamic_obs = self.get_dynamic_obstacles(world)
        obs = agents + static_obs + dynamic_obs
        # penalty of collision with agents
        for i, o in enumerate(obs):
            if o != agent:
                if self.is_collision(agent, o):
                    if not agent.collide:
                        agent.crash += 1
                        agent.collide = True
                    break
            if i == len(obs) - 1:
                agent.collide = False
        # min of all ray's detect result
        idx = 0
        obs_dis = []
        for i in range(len(agent.ray)):
            obs_dis.append(agent.ray[i][0])
        agt_dis = []
        agt_ang = []
        for entity in agents:
            if entity != agent:
                dis2agt = np.array([min(norm(entity.state.p_pos - agent.state.p_pos) / 100, 50 * agent.size / 100)])
                ang = self.get_relAngle(agent, entity)
                agt_dis.append(dis2agt)
                agt_ang.append(np.array([ang]))

        #rew -= alpha * math.exp(-beta * min(agt_dis))
        stream_err = self.get_stream_value(agent)
        formation_err = norm(agent.err)
        # only when the desire formation position is in front of the agent (0~pi) should it consider the stream line following
        if formation_err < 1 and abs(self.wrap2pi(agent.state.p_ang - world.agents[0].state.p_ang - target_ang)) > np.pi/4:
            rew -= alpha * math.exp(-beta * min(min(obs_dis), min(agt_dis)))
        else:
            rew -= alpha * stream_err

        # reward for formation
        c = 1
        #rew -= c*(50*(agent.err - agent.err_prev))
        rew -= c*formation_err
        return rew

    def leader_reward(self, agent, world):
        # Agent are rewarded based on minimum agent distance to landmark
        rew = -0.3
        agents = self.follower_agents(world)
        leader = agent
        static_obs = self.get_static_obstacles(world)
        dynamic_obs = self.get_dynamic_obstacles(world)
        obs = static_obs + dynamic_obs
        alpha = 5
        beta = 2

        for o in obs:
            if self.is_collision(agent, o):
                #rew -= 5
                pass
        idx = 0
        for i in range(len(agent.ray)):
            if agent.ray[i][0] < agent.ray[idx][0]:
                idx = i
        agt_dis = []
        agt_ang = []
        for entity in world.agents:
            if entity != agent:
                dis2agt = np.array([min(norm(entity.state.p_pos - agent.state.p_pos) / 100, 50 * agent.size / 100)])
                ang = self.get_relAngle(agent, entity)
                agt_dis.append(dis2agt)
                agt_ang.append(np.array([ang]))
        rew -= alpha * math.exp(-beta * min(agent.ray[idx][0], min(agt_dis)))
        # leader's task is to navigate toward the goal
        sigma = 100
        rew -= sigma*(leader.dis2goal - leader.dis2goal_prev)*abs(np.cos(leader.ang2goal))
        return rew

    def kalman_filter(self, world, agent):
        '''
        kalman filter for followers to predict leader trajectory
        leader_state = [pos_x,pos_y,vel_x,vel_y]
        follower_obs = [pos_x,pos_y]
        his_size = 10
        :return: predicted leader state
        '''
        x = agent.dis2leader*np.cos(agent.ang2leader)
        y = agent.dis2leader*np.sin(agent.ang2leader)
        v_x = 0
        v_y = 0
        if len(agent.his):
            v_x = x - agent.his[-1][2]
            v_y = y - agent.his[-1][3]
        state_cur = np.array([x,y,v_x,v_y]).T
        agent.his.append(state_cur)
        det = world.dt
        Z = np.array([x,y]).T
        F = np.array([[1,0,det,0],[0,1,0,det],[0,0,1,0],[0,0,0,1]])
        H = np.array([[1,0,0,0],[0,1,0,0]])
        R = 0
        if len(agent.his):
            R = np.cov(np.array(agent.his).T)

        state_pre = F*state_cur
        P_pre = F*agent.P*F.T
        K = P_pre*H.T*np.invert(H*P_pre*H.T+R)
        state_new = state_pre + K*(Z-H*state_pre)
        agent.P = (np.eye(4)-K*H)*P_pre
        return state_new

    def observation2(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)

    def observation(self, agent, world):
        # update desire center position
        p_pos = agent.state.p_pos

        if agent.leader:
            agent.dis2goal_prev = agent.dis2goal
            agent.dis2goal = norm(p_pos - world.landmarks[0].state.p_pos) / 100
            agent.ang2goal = self.get_relAngle(agent, world.landmarks[0])
        else:
            leader = self.leader_agents(world)
            agent.dis2leader_prev = agent.dis2leader
            agent.dis2leader = norm(p_pos - leader[0].state.p_pos) / 100
            agent.ang2leader = self.get_relAngle(leader[0], agent)
            agent.p_rel = agent.dis2leader * np.array([math.cos(agent.ang2leader), math.sin(agent.ang2leader)])

        # get distance and relative angle of all entities in this agent's reference frame
        agt_dis = []
        agt_ang = []
        obs_dis = []
        obs_ang = []
        obs_size = []
        obs_type = []
        obs = world.agents + world.static_obs + world.dynamic_obs
        for entity in obs:
            if 'dynamic_obs' in entity.name:
                entity.state.p_vel[0] = min(0.3, max(0.1,
                                                     entity.state.p_vel[0] + 0.01 * np.random.choice([0, 1], size=1,
                                                                                                     p=[.97, .03])))
            dis2obs = (norm(entity.state.p_pos - p_pos)-entity.size-agent.size)/100
            ang = self.get_relAngle(agent, entity)
            if dis2obs < 25*agent.size/100 and entity != agent:
                obs_dis.append(dis2obs)
                obs_ang.append(ang)
                obs_size.append(entity.size/100)
                obs_type.append(entity.name)
            if 'agent' in entity.name and entity != agent:
                dis2agt = np.array([min(norm(entity.state.p_pos - p_pos)/100, 50*agent.size/100)])
                ang = self.get_relAngle(agent, entity)
                agt_dis.append(dis2agt)
                agt_ang.append(np.array([ang]))
        self.set_agent_ray(agent, obs_dis, obs_ang, obs_type)
        sensor_ray = [np.array([agent.ray[i][j]]) for i in range(len(agent.ray)) for j in range(len(agent.ray[i]))]

        # communication of all other agents, now assume the communication graph is fully connected
        formation_pos_x = []
        formation_pos_y = []
        formation_pos_x.append(p_pos[0])
        formation_pos_y.append(p_pos[1])
        comm = [world.agents[0].state.c]
        for other in world.agents:
            if other is agent: continue
            formation_pos_x.append(other.state.p_pos[0])
            formation_pos_y.append(other.state.p_pos[1])
            agent.agents_ctr = np.array([np.mean(formation_pos_x), np.mean(formation_pos_y)])

        vel = []
        omg = []
        ang = []
        if not agent.leader:
            ang.append(np.array([self.wrap2pi(agent.ang2leader-agent.w_des) / np.pi]))
        else:
            ang.append(np.array([agent.ang2goal / np.pi]))
        vel.append(np.array([norm(agent.state.p_vel)/100]))
        omg.append(np.array([agent.state.p_omg / np.pi]))
        err = [np.array([agent.err[i]]) for i in range(len(agent.err))]
        if agent.leader:
            return np.concatenate([np.array([100*(agent.dis2goal - agent.dis2goal_prev)])] + ang + vel + omg + agt_dis + agt_ang + sensor_ray)
            #return np.concatenate([np.array([100 * (agent.dis2goal - agent.dis2goal_prev)])] + ang + vel + omg + sensor_ray)
        else:
            #return np.concatenate([np.array([agent.dis2leader - agent.d_des])] + ang + vel + omg + agt_dis + agt_ang + sensor_ray)
            return np.concatenate(err + vel + omg + agt_dis + agt_ang + sensor_ray)

    def constraint(self, agent, world):
        return []

    def done(self, agent, world):
        #return not np.any(world.err > 2e-2)
        return world.agents[0].dis2goal < 5e-1


