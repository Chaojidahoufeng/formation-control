#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse

from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    #parser.add_argument('-s', '--scenario', default='formation_tracking.py', help='Path of the scenario Python script.')
    parser.add_argument('-s', '--scenario', default='formation_stream_avoidance_4.py', help='Path of the scenario Python script.')
    parser.add_argument("--agent_num", type=int, default=4, help="number of agents")
    parser.add_argument("--static_obstacle_num", type=int, default=5, help="number of static obstacles")
    parser.add_argument("--dynamic_obstacle_num", type=int, default=5, help="number of dynamic obstacles")
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world(args)
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.constraint, done_callback=scenario.done, info_callback=None, shared_viewer=True)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    policies = [InteractivePolicy(env, i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    while True:
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        #print(act_n)
        # step environment
        obs_n, formation_n, avoidance_n, done_n, _, _= env.step(act_n)
        # render all agent views
        env.render()
        # display rewards
        #for agent in env.world.agents:
        #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
