import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
from matplotlib import pyplot as plt

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
from maddpg.trainer.ddpg import DDPGAgentTrainer
from maddpg.trainer.cddpg import CDDPGAgentTrainer
from maddpg.trainer.dqn import DQNAgentTrainer
import tensorflow.contrib.layers as layers



def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="rel_based_formation_stream_avoidance_4", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=250, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=30000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--network", type=str, default="MLP", help="define neural network type")
    parser.add_argument("--trajectory_size", type=int, default=25)
    parser.add_argument("--apply_noise", type=bool, default=True)
    parser.add_argument("--noise_type", type=str, default="adaptive-param_0.2")
    parser.add_argument("--param_noise_adaption_interval", type=int, default=50)

    # Checkpointin1
    parser.add_argument("--exp-name", type=str, default='Test_benchmark', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="../policy/model_benchmark.ckpt", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="../trainResult/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="../trainResult/", help="directory where plot data is saved")

    # world setting
    parser.add_argument("--map-max-size", type=int, default=1200)
    parser.add_argument("--agent-init-bound", type=int, default=200)
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None, is_training=True):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        def bn_layer(x, epsilon=0.001, decay=0.9, reuse=None, name=None):
            """
            Performs a batch normalization layer
            Args:
                x: input tensor
                scope: scope name
                is_training: python boolean value
                epsilon: the variance epsilon - a small float number to avoid dividing by 0
                decay: the moving average decay
            Returns:
                The ops of a batch normalization layer
            """
            shape = x.get_shape().as_list()
            # gamma: a trainable scale factor

            #gamma = tf.get_variable("gamma"+"_"+name, shape[-1], initializer=tf.constant_initializer(1.0), trainable=True)
            gamma = tf.Variable(tf.ones(shape[-1]), trainable=True, name="gamma"+"_"+name)
            # beta: a trainable shift value
            #beta = tf.get_variable("beta"+"_"+name, shape[-1], initializer=tf.constant_initializer(0.0), trainable=True)
            beta = tf.Variable(tf.zeros(shape[-1]), trainable=True, name="beta" + "_" + name)
            #avg = tf.get_variable("avg"+"_"+name, shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
            avg = tf.Variable(tf.zeros(shape[-1]), trainable=True, name="avg" + "_" + name)
            #moving_avg = tf.get_variable("moving_avg"+"_"+name, shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
            moving_avg = tf.Variable(tf.zeros(shape[-1]), trainable=True, name="moving_avg" + "_" + name)
            #var = tf.get_variable("var" + "_" + name, shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
            var = tf.Variable(tf.ones(shape[-1]), trainable=True, name="var" + "_" + name)
            #moving_var = tf.get_variable("moving_var"+"_"+name, shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
            moving_var = tf.Variable(tf.ones(shape[-1]), trainable=True, name="moving_var" + "_" + name)
            if is_training:
                # tf.nn.moments == Calculate the mean and the variance of the tensor x
                temp_avg, temp_var = tf.nn.moments(x, np.arange(len(shape) - 1), keep_dims=True)
                temp_avg = tf.reshape(temp_avg, [temp_avg.shape.as_list()[-1]])
                temp_var = tf.reshape(temp_var, [temp_var.shape.as_list()[-1]])
                update_avg = tf.assign(avg, temp_avg)
                update_var = tf.assign(var, temp_var)
                # update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
                update_moving_avg = tf.assign(moving_avg, moving_avg.value() * decay + avg.value() * (1 - decay))
                # update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
                update_moving_var = tf.assign(moving_var, moving_var.value() * decay + var.value() * (1 - decay))
                control_inputs = [update_avg, update_var, update_moving_avg, update_moving_var]
            else:
                avg = moving_avg
                var = moving_var
                control_inputs = []
            with tf.control_dependencies(control_inputs):
                output = tf.nn.relu(tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon))

            return output
        def bn_layer_top(x, epsilon=0.001, decay=0.99, name=None):
            """
            Returns a batch normalization layer that automatically switch between train and test phases based on the
            tensor is_training
            Args:
                x: input tensor
                scope: scope name
                is_training: boolean tensor or variable
                epsilon: epsilon parameter - see batch_norm_layer
                decay: epsilon parameter - see batch_norm_layer
            Returns:
                The correct batch normalization layer based on the value of is_training
            """
            # assert isinstance(is_training, (ops.Tensor, variables.Variable)) and is_training.dtype == tf.bool

            '''return tf.cond(
                is_training,
                lambda: bn_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=True, reuse=None),
                lambda: bn_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=False, reuse=True),
            )'''
            return bn_layer(x=x, epsilon=epsilon, decay=decay, reuse=None, name=name) if is_training else bn_layer(x=x, epsilon=epsilon, decay=decay, reuse=True, name=name)
            # out = layers.batch_norm(input, is_training=is_training)
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units*2, activation_fn=tf.nn.relu)
        #out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        if "p_func" in scope:
            out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        else:
            out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
            #out = tf.matmul(out, out, transpose_b=True)
        return out

def lstm_model(obs, trajectory_size, num_inputs, num_outputs, scope, reuse=False, num_units=300, rnn_cell=None):

    obs = tf.reshape(obs, [-1, trajectory_size, num_inputs])
    sequence = tf.placeholder(tf.float32, [None, trajectory_size, num_inputs])
    def length(sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def linear(input):
        out = tf.reshape(input, [-1, num_inputs])
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=128, activation_fn=tf.nn.relu)
        #out = layers.fully_connected(out, num_outputs=300, activation_fn=tf.nn.relu)
        #out = tf.reshape(out, [-1, trajectory_size, 128])
        return out
    outputs = []

    with tf.variable_scope(scope, reuse=reuse):
        softmax_w = tf.get_variable(name="weight", shape=[19, num_outputs])
        softmax_b = tf.get_variable(name="bias", shape=[num_outputs])
        for step in range(trajectory_size):
            outputs.append(obs[:, step, :])
        outputs = tf.split(tf.concat(outputs,1), 19, 1)
        output, state = tf.contrib.rnn.static_rnn(tf.contrib.rnn.BasicLSTMCell(19), outputs, dtype=tf.float32)
        return tf.matmul(output[-1], softmax_w) + softmax_b

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(arglist)
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.constraint, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.constraint, done_callback=scenario.done)
    return env

def get_trainers(env, num_adversaries, his_shape_n, obs_shape_n, arglist):
    trainers = []
    if arglist.network=="MLP":
        model = mlp_model
        if arglist.good_policy=="ddpg":
            trainer = DDPGAgentTrainer
            for i in range(2):
                # the last trainer is for saving all followers' transitions
                trainers.append(trainer("agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist))
        elif arglist.good_policy=="dqn":
            trainer = DQNAgentTrainer
            trainers.append(trainer(model, obs_shape_n, [env.action_space[0]], env.n, arglist,
                                    local_q_func=(arglist.adv_policy == 'ddpg')))
        elif arglist.good_policy=="cddpg":
            trainer = CDDPGAgentTrainer
            for i in range(2):
                # the last trainer is for saving all followers' transitions
                trainers.append(trainer("agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist))
        else:
            trainer = MADDPGAgentTrainer
            for i in range(num_adversaries):
                trainers.append(trainer(
                    "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
                    local_q_func=(arglist.adv_policy=='ddpg')))
            for i in range(num_adversaries, env.n):
                trainers.append(trainer(
                    "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
                    local_q_func=(arglist.good_policy=='ddpg')))
    return trainers

def render_rel_position(figure, ax, agent_idx, obs):
    agt_dis = obs[4:7]
    agt_ang = obs[7:10]
    agent_0_pos = [0.,0.]
    other_agent_pos = [ [agt_dis[i]*np.cos(agt_ang[i]), agt_dis[i]*np.sin(agt_ang[i])] for i in range(len(agt_dis)) ]

    ax[agent_idx].set_title("Agent-"+str(agent_idx), family='sans-serif',
                    fontname='Helvetica',
                    fontsize=20)
    for _, pos in enumerate(other_agent_pos):
        ax[agent_idx].scatter(pos[0], pos[1], color='r')

    ax[agent_idx].scatter(agent_0_pos[0], agent_0_pos[1], color='g')

    for _ in range(len(ax)):
        plt.tight_layout()

    plt.gcf().canvas.flush_events()
    figure.canvas.start_event_loop(0.001)
    plt.gcf().canvas.flush_events()



def train(arglist):
    # rendering setting
    if arglist.display:
        plt.ion()
        figure, ax = plt.subplots(1, 4, figsize=(4*4, 4),
                                                    facecolor="whitesmoke",
                                                    num="Thread")
    #tf.reset_default_graph()
    with U.single_threaded_session():
        tf.set_random_seed(0)
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        if arglist.good_policy == "ddpg" or arglist.good_policy == "cddpg" :
            obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
            his_shape_n = [((env.observation_space[i].shape[0]+3)*arglist.trajectory_size,) for i in range(env.n)]
        else:
            obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
            his_shape_n = [((env.observation_space[i].shape[0]+3)*arglist.trajectory_size,) for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        episode_step = [0]
        final_ep_steps = []
        episode_done = [0]
        final_ep_done = []
        train_step = 0
        trainers = get_trainers(env, num_adversaries, his_shape_n, obs_shape_n, arglist)
        #vl = [v for v in tf.global_variables() if "Adam" not in v.name]
        #saver = tf.train.Saver(var_list=vl)
        saver = tf.train.Saver()

        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir, saver)

        episode_rewards = [0.0]  # sum of rewards for all agents
        episode_constraints = [0.0]  # sum of constraints for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        agent_constraints = [[0.0] for _ in range(env.n)]  # individual agent constraint
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        final_ep_constraints = []  # sum of constraints for training curve
        final_ep_ag_constraints = []  # agent constraints for training curve
        episode_crash = [0]  # sum of crashes for all agents
        final_ep_crash = []  # sum of crashes for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info

        obs_n = env.reset()
        t_start = time.time()
        final_reward_prev = None
        print('Starting iterations...')

        while True:
            if arglist.network == "MLP":
                # get action
                if arglist.good_policy == "maddpg":
                    action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                elif arglist.good_policy == "ddpg" or arglist.good_policy == "cddpg":
                    action_n = [trainers[obs if obs == 0 else -1].action(obs_n[obs], len(episode_rewards)) for obs in range(len(obs_n))]
                    constraint_n = [trainers[obs if obs == 0 else -1].constraint(obs_n[obs]) for obs in range(len(obs_n))]
                # environment step
                new_obs_n, navigation_reward_n, avoidance_reward_n, formation_reward_n, done_n, info_n, crash_n = env.step(action_n)

                rew_n = [navigation_reward_n[i]+avoidance_reward_n[i]+formation_reward_n[i] for i in range(len(navigation_reward_n))]
                episode_step[-1] += 1
                done = all(done_n)
                terminal = (episode_step[-1] >= arglist.max_episode_len)
                # collect experience
                if arglist.good_policy == "ddpg":
                    for i in range(len(obs_n)):
                        trainers[i!=0].experience(obs_n[i], action_n[i], rew_n[i] - constraint_n[i], new_obs_n[i], done_n[i], terminal)
                elif arglist.good_policy == "cddpg":
                    for i in range(len(obs_n)):
                        trainers[i].experience(obs_n[i], action_n[i], constraint_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
                        if i > 0:
                            trainers[-1].experience(obs_n[i], action_n[i], constraint_n[i], rew_n[i], new_obs_n[i],
                                                   done_n[i], terminal)
                else:
                    for i, agent in enumerate(trainers):
                        agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
                obs_n = new_obs_n
                for i, rew in enumerate(rew_n):
                    # ignore leader reward
                    episode_rewards[-1] += rew
                    agent_rewards[i][-1] += rew

                # for i, con in enumerate(constraint_n):
                #     if i > 0:
                #         episode_constraints[-1] += con[0]
                #         agent_constraints[i][-1] += con[0]

                for i, crash in enumerate(crash_n):
                    episode_crash[-1] += crash

                for i, done in enumerate(done_n):
                    episode_done[-1] += done

                if done or terminal:
                    obs_n = env.reset()
                    episode_step.append(0)
                    episode_rewards.append(0)
                    episode_constraints.append(0)
                    episode_crash.append(0)
                    episode_done.append(0)
                    for a in agent_rewards:
                        a.append(0)
                    agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.01)
                env.render()


                for i in range(len(trainers)):
                    ax[i].clear()
                    # ax[i].set_yticks([])
                    # ax[i].set_xticks([])
                    # ax[i].set_yticklabels([])
                    # ax[i].set_xticklabels([])
                for i in range(len(trainers)):
                    render_rel_position(figure, ax, i, obs_n[i])
                '''if train_step == 10:
                    break'''
                # continue

            if arglist.good_policy == "ddpg" or arglist.good_policy == "cddpg":
                for i in range(len(obs_n)):
                    # update all trainers, if not in display or benchmark mode
                    trainers[i if i == 0 else -1].preupdate()
                    loss = trainers[i if i == 0 else -1].update(trainers, train_step)
            else:
                for agent in trainers:
                    loss = agent.update(trainers, train_step)
            # save model, display training output
            if (done or terminal) and (len(episode_rewards) % arglist.save_rate == 0):
                final_reward = np.mean(episode_rewards[-arglist.save_rate:])
                final_constraint = np.mean(episode_constraints[-arglist.save_rate:])
                final_step = np.mean(episode_step[-arglist.save_rate:])
                final_crash = np.mean(episode_crash[-arglist.save_rate:])
                final_done = np.mean(episode_done[-arglist.save_rate:])
                if not final_reward_prev:
                    final_reward_prev = final_reward
                else:
                    if final_reward > final_reward_prev:
                        U.save_state(arglist.save_dir, saver=saver)
                        final_reward_prev = final_reward
                        print("model saved time: ", time.asctime(time.localtime(time.time())))
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, mean episode step: {},  mean episode crash: {}, time: {}".format(
                        train_step, len(episode_rewards), final_reward, final_step, final_crash, round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), final_reward,
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(final_reward)
                final_ep_constraints.append(final_constraint)
                final_ep_steps.append(final_step)
                final_ep_crash.append(final_crash)
                final_ep_done.append(final_done)
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))
                for con in agent_constraints:
                    final_ep_ag_constraints.append(np.mean(con[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                step_file_name = arglist.plots_dir + arglist.exp_name + '_steps.pkl'
                with open(step_file_name, 'wb') as fp:
                    pickle.dump(final_ep_steps, fp)
                crash_file_name = arglist.plots_dir + arglist.exp_name + '_crashes.pkl'
                with open(crash_file_name, 'wb') as fp:
                    pickle.dump(final_ep_crash, fp)
                done_file_name = arglist.plots_dir + arglist.exp_name + '_done.pkl'
                with open(done_file_name, 'wb') as fp:
                    pickle.dump(final_ep_done, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
    '''graph = tf.get_default_graph()
    sess = tf.Session()
    saver = tf.train.import_meta_graph("/home/shinyochiu/maddpg/policy/model.ckpt.meta")
    saver.restore(sess, "/home/shinyochiu/maddpg/policy/model.ckpt")
    tf.train.write_graph(sess.graph_def, '.', '/home/shinyochiu/maddpg/policy/graph.pb', as_text=False)
    converter = tf.lite.TFLiteConverter.from_saved_model("/home/shinyochiu/maddpg/policy/graph.pb")
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)'''

    '''import pprint, pickle
    import matplotlib.pyplot as plt
    rew1_file_name = arglist.plots_dir + arglist.exp_name + '_rewards_formation_navigation_1.pkl'
    pkl_file1 = open(rew1_file_name, 'rb')
    rew2_file_name = arglist.plots_dir + arglist.exp_name + '_rewards_formation_navigation_5.pkl'
    pkl_file2 = open(rew2_file_name, 'rb')
    step_file_name = arglist.plots_dir + arglist.exp_name + '_steps_formation_navigation_5.pkl'
    pkl_file5 = open(step_file_name, 'rb')
    rew1 = pickle.load(pkl_file1)
    rew2 = pickle.load(pkl_file2)
    steps = pickle.load(pkl_file5)
    print(steps)
    #plt.plot(rew1[0:len(steps)], label='MADDPG')
    plt.plot(rew2[0:len(steps)], label='rewards')
    plt.legend()
    plt.xlabel('episode (x1000)')
    plt.ylabel('rewards')
    plt.show()'''
