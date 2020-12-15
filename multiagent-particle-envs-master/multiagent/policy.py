import numpy as np
from pyglet.window import key

# individual agent policy
class Policy(object):
    def __init__(self):
        pass
    def action(self, obs):
        raise NotImplementedError()

# interactive policy based on keyboard input
# hard-coded to deal only with movement, not communication
class InteractivePolicy(Policy):
    def __init__(self, env, agent_index):
        super(InteractivePolicy, self).__init__()
        self.env = env
        # hard-coded keyboard events
        self.move = [False for _ in range(3)]
        self.comm = [False for _ in range(env.world.dim_c)]
        self.agent_idx = agent_index
        #self.move[2] = True
        # register keyboard events with this environment's window
        env.viewers[0].window.on_key_press = self.key_press
        env.viewers[0].window.on_key_release = self.key_release

    def action(self, obs):
        # ignore observation and just act based on keyboard events

        if self.env.discrete_action_input:
            u = 0
            if self.move[0]: u = 1
            if self.move[1]: u = 2
            if self.move[2]: u = 4
            if self.move[3]: u = 3
        else:
            #u = np.zeros(5) # 5-d because of no-move action
            self.u = np.zeros(3)
            if self.move[0]:
                # turn right
                self.u[0] += 1.0
                #self.move[0] = False
                #if self.u[2] != 0 or self.u[3] != 0:
                '''if self.u[1] == 0:
                    self.move[1] = False
                    self.u[0] += 1.0
                else:
                    self.u[1] -= 1.0'''
            if self.move[1]:
                # turn left
                self.u[1] += 1.0
                #self.move[1] = False
                #if self.u[2] != 0 or self.u[3] != 0:
                '''if self.u[0] == 0:
                    self.move[0] = False
                    self.u[1] += 1.0
                else:
                    self.u[0] -= 1.0'''
            if self.move[2]:
                # move forward
                self.u[2] += 1.0
                
            '''if self.move[3]:
                # move backward
                self.u[3] += 1.0'''

            '''if True not in self.move:
                self.u[0] += 1.0'''
        return np.concatenate([self.u, np.zeros(self.env.world.dim_c)])

    # keyboard event callbacks # 20190717 modify control policy, each time pressing keyboard will accelerate the speed
    def key_press(self, k, mod):
        if k==key.RIGHT:
            self.move[0] = True
        if k==key.LEFT:
            self.move[1] = True
        if k==key.UP:
            self.move[2] = True
        '''if k==key.DOWN:
            self.move[3] = True'''

    def key_release(self, k, mod):
        if k==key.LEFT:  self.move[1] = False
        if k==key.RIGHT: self.move[0] = False
        if k==key.UP:    self.move[2] = False
        #if k==key.DOWN:  self.move[3] = False
