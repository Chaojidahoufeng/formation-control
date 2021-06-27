import argparse
import numpy as np
import tensorflow as tf
import time
import pickle

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
from maddpg.trainer.ddpg import DDPGAgentTrainer
from maddpg.trainer.cddpg import CDDPGAgentTrainer
from maddpg.trainer.dqn import DQNAgentTrainer
import tensorflow.contrib.layers as layers

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--id", type=str, default="1", help="experiment id")
    parser.add_argument("--scenario", type=str, default="formation_stream_avoidance_", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=250, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=30000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="ddpg", help="policy for good agents")
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
    parser.add_argument("--exp-name", type=str, default='Test_', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="../policy/model_", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--monte-carlo", action="store_true", default=False)
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="../trainResult/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="../trainResult/", help="directory where plot data is saved")
    # World setting
    parser.add_argument("--agent_num", type=int, default=4, help="number of agents")
    parser.add_argument("--static_obstacle_num", type=int, default=5, help="number of static obstacles")
    parser.add_argument("--dynamic_obstacle_num", type=int, default=5, help="number of dynamic obstacles")
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
        if arglist.good_policy == "ddpg":
            trainer = DDPGAgentTrainer
            for i in range(2):
                # the last trainer is for saving all followers' transitions
                trainers.append(trainer("agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist))
        elif arglist.good_policy == "ddpg2":
            trainer = DDPG2AgentTrainer
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


def train(arglist):
    #tf.reset_default_graph()
    with U.single_threaded_session():
        tf.set_random_seed(0)
        # Create environment
        env = make_env(arglist.scenario + arglist.id, arglist, arglist.benchmark)
        # Create agent trainers
        if "ddpg" in arglist.good_policy or arglist.good_policy == "cddpg" :
            obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
            his_shape_n = [((env.observation_space[i].shape[0]+3)*arglist.trajectory_size,) for i in range(env.n)]
        else:
            obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
            his_shape_n = [((env.observation_space[i].shape[0]+3)*arglist.trajectory_size,) for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        episode_step = [0]
        ep_steps = []
        episode_done = [0]
        ep_done = []
        step = 0
        trainers = get_trainers(env, num_adversaries, his_shape_n, obs_shape_n, arglist)
        #vl = [v for v in tf.global_variables() if "Adam" not in v.name]
        #saver = tf.train.Saver(var_list=vl)
        saver = tf.train.Saver()

        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        state = '_train'
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir + arglist.id + '.ckpt'
        if arglist.display or arglist.restore or arglist.benchmark or arglist.monte_carlo:
            print('Loading previous state...')
            state = '_eval' 
            U.load_state(arglist.load_dir, saver)

        episode_formation_err = [0.0]  # sum of rewards for all agents
        episode_avoidance_err = [0.0]  # sum of rewards for all agents
        episode_constraints = [0.0]  # sum of constraints for all agents
        agent_formation_err = [[0.0] for _ in range(env.n)]  # individual agent reward
        agent_avoidance_err = [[0.0] for _ in range(env.n)]  # individual agent reward
        agent_constraints = [[0.0] for _ in range(env.n)]  # individual agent constraint
        ep_formation_err = []  # sum of rewards for training curve
        ep_avoidance_err = []  # sum of rewards for training curve
        ep_ag_formation_err = []  # agent rewards for training curve
        ep_ag_avoidance_err = []  # agent rewards for training curve
        ep_constraints = []  # sum of constraints for training curve
        ep_ag_constraints = []  # agent constraints for training curve
        episode_crash = [0]  # sum of crashes for all agents
        ep_crash = []  # sum of crashes for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        
        display_formation_err = [[] for _ in range(env.n)]
        obs_n = env.reset()
        t_start = time.time()
        final_reward_prev = None
        print('Starting iterations...')

        while True:
            if arglist.network == "MLP":
                # get action
                if arglist.good_policy == "maddpg":
                    action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                elif "ddpg" in arglist.good_policy or arglist.good_policy == "cddpg":
                    action_n = [trainers[obs if obs == 0 else -1].action(obs_n[obs], len(episode_formation_err)) for obs in range(len(obs_n))]
                    constraint_n = [trainers[obs if obs == 0 else -1].constraint(obs_n[obs]) for obs in range(len(obs_n))]
                # environment step
                new_obs_n, formation_rew_n, avoidance_rew_n, done_n, info_n, crash_n = env.step(action_n)
                episode_step[-1] += 1
                done = all(done_n)
                # collect experience
                terminal = (episode_step[-1] >= arglist.max_episode_len)

                if state == '_train':
                    if "ddpg" in arglist.good_policy:
                        for i in range(len(obs_n)):
                            trainers[i!=0].experience(obs_n[i], action_n[i], formation_rew_n[i] + avoidance_rew_n[i], new_obs_n[i], done_n[i], terminal)
                    elif arglist.good_policy == "cddpg":
                        for i in range(len(obs_n)):
                            trainers[i].experience(obs_n[i], action_n[i], constraint_n[i], formation_rew_n[i] + avoidance_rew_n[i], new_obs_n[i], done_n[i], terminal)
                            if i > 0:
                                trainers[-1].experience(obs_n[i], action_n[i], constraint_n[i], formation_rew_n[i] + avoidance_rew_n[i], new_obs_n[i], done_n[i], terminal)
                    else:
                        for i, agent in enumerate(trainers):
                            agent.experience(obs_n[i], action_n[i], formation_rew_n[i] + avoidance_rew_n[i], new_obs_n[i], done_n[i], terminal)

                    if "ddpg" in arglist.good_policy or arglist.good_policy == "cddpg":
                        for i in range(len(obs_n)):
                            # update all trainers, if not in display or benchmark mode
                            trainers[i if i == 0 else -1].preupdate()
                            loss = trainers[i if i == 0 else -1].update(trainers, step)
                    else:
                        for agent in trainers:
                            loss = agent.update(trainers, train_step)

                obs_n = new_obs_n
                for i, rew in enumerate(formation_rew_n):
                    # ignore leader reward
                    if i > 0:
                        episode_formation_err[-1] += rew
                        agent_formation_err[i][-1] += rew

                for i, rew in enumerate(avoidance_rew_n):
                    if i > 0:
                        episode_avoidance_err[-1] += rew
                        agent_avoidance_err[i][-1] += rew


                for i, crash in enumerate(crash_n):
                    if i > 0:
                        episode_crash[-1] += crash

                for i, done in enumerate(done_n):
                    if i > 0:
                        episode_done[-1] += done

            # increment global step counter
            step += 1

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.01)
                env.render()
                for i in range(env.n):
                    display_formation_err[i].append(-formation_rew_n[i])
                if (done or terminal):
                    import pprint, pickle
                    import matplotlib.pyplot as plt
                    display_formation_err_file_name = arglist.plots_dir + arglist.exp_name + arglist.id + state + '_display_formation_err.pkl'
                    with open(display_formation_err_file_name, 'wb') as fp:
                        pickle.dump(display_formation_err, fp)
                    break
                continue

            # save model, display training output
            if (done or terminal) and (len(episode_formation_err) % arglist.save_rate == 0):
                formation_err = np.mean(episode_formation_err[-arglist.save_rate:])
                avoidance_err = np.mean(episode_avoidance_err[-arglist.save_rate:])
                step = np.mean(episode_step[-arglist.save_rate:])
                crash = np.mean(episode_crash[-arglist.save_rate:])
                done = np.mean(episode_done[-arglist.save_rate:])

                if not final_reward_prev:
                    final_reward_prev = formation_err + avoidance_err
                else:
                    final_reward = formation_err + avoidance_err
                    if final_reward > final_reward_prev:
                        U.save_state(arglist.save_dir + arglist.id + '.ckpt', saver=saver)
                        final_reward_prev = final_reward
                        print("model saved time: ", time.asctime(time.localtime(time.time())))
                # print statement depends on whether or not there are adversaries
                print("steps: {}, episodes: {}, mean episode formation error: {}, mean episode avoidance error: {}, mean episode step: {},  mean episode crash: {}, time: {}".format(step, len(episode_formation_err), formation_err, avoidance_err, step, crash, round(time.time()-t_start, 3)))
                print(action_n)

                t_start = time.time()
                # Keep track of final episode reward
                ep_formation_err.append(formation_err)
                ep_avoidance_err.append(avoidance_err)
                ep_steps.append(step)
                ep_crash.append(crash)
                ep_done.append(done)

                for rew in agent_formation_err:
                    ep_ag_formation_err.append(np.mean(rew[-arglist.save_rate:]))
                for rew in agent_avoidance_err:
                    ep_ag_avoidance_err.append(np.mean(rew[-arglist.save_rate:]))


            if done or terminal:
                obs_n = env.reset()
                episode_step.append(0)
                episode_formation_err.append(0)
                episode_avoidance_err.append(0)

                episode_crash.append(0)
                episode_done.append(0)
                for a in agent_formation_err:
                    a.append(0)
                for a in agent_avoidance_err:
                    a.append(0)
                agent_info.append([[]])

            # saves final episode reward for plotting training curve later
            if len(episode_formation_err) > arglist.num_episodes:   
                import pprint, pickle
                import matplotlib.pyplot as plt
                formation_err_file_name = arglist.plots_dir + arglist.exp_name + arglist.id + state + '_formation_err.pkl'
                with open(formation_err_file_name, 'wb') as fp:
                    pickle.dump(ep_formation_err, fp)
                agformation_err_file_name = arglist.plots_dir + arglist.exp_name + arglist.id + state + '_agformation_err.pkl'
                with open(agformation_err_file_name, 'wb') as fp:
                    pickle.dump(ep_ag_formation_err, fp)
                avoidance_err_file_name = arglist.plots_dir + arglist.exp_name + arglist.id + state + '_avoidance_err.pkl'
                with open(avoidance_err_file_name, 'wb') as fp:
                    pickle.dump(ep_avoidance_err, fp)
                agavoidance_err_file_name = arglist.plots_dir + arglist.exp_name + arglist.id + state + '_agavoidance_err.pkl'
                with open(agavoidance_err_file_name, 'wb') as fp:
                    pickle.dump(ep_ag_avoidance_err, fp)
                step_file_name = arglist.plots_dir + arglist.exp_name + arglist.id + state + '_steps.pkl'
                with open(step_file_name, 'wb') as fp:
                    pickle.dump(ep_steps, fp)
                crash_file_name = arglist.plots_dir + arglist.exp_name + arglist.id + state + '_crashes.pkl'
                with open(crash_file_name, 'wb') as fp:
                    pickle.dump(ep_crash, fp)
                done_file_name = arglist.plots_dir + arglist.exp_name + arglist.id + state + '_done.pkl'
                with open(done_file_name, 'wb') as fp:
                    pickle.dump(ep_done, fp)
                print('...Finished total of {} episodes.'.format(len(episode_formation_err)))
                break

if __name__ == '__main__':
    import pprint, pickle
    import matplotlib.pyplot as plt
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
    '''
    rew1_file_name = arglist.plots_dir + arglist.exp_name + '1' + '_eval_display_formation_err.pkl'
    rew1_file = open(rew1_file_name, 'rb')
    rew2_file_name = arglist.plots_dir + arglist.exp_name + '2' + '_eval_display_formation_err.pkl'
    rew2_file = open(rew2_file_name, 'rb')
    rew3_file_name = arglist.plots_dir + arglist.exp_name + '4' + '_eval_display_formation_err.pkl'
    rew3_file = open(rew3_file_name, 'rb')

    rew1 = np.array(pickle.load(rew1_file))
    rew2 = np.array(pickle.load(rew2_file))
    rew3 = np.array(pickle.load(rew3_file))
    plt.figure(figsize=(10,8))
    labels = ['$APF$ with $d_{risk}$','$APF$ with $d_{stop}$','Stream-based']
    for i in range(0,4):
        plt.subplot(221+i)
        plt.title('Agent '+str(i+1),fontsize=1)
        plt.plot(rew1[i+1,0:], '-.', label=labels[0], linewidth=1)
        plt.plot(rew2[i+1,0:], '--', label=labels[1], linewidth=1)
        plt.plot(rew3[i+1,0:], '-', label=labels[2], linewidth=1)
        plt.xlabel('Steps (0.1 sec/step)', fontsize=14)
        plt.ylabel('tracking errors (m/step)', fontsize=14)
        plt.xlim(0,2500)
        plt.ylim(0,3.0)
    plt.legend(labels=labels, bbox_to_anchor=(-0.15, -0.4), loc="lower center", fontsize=12, ncol = 3)
    plt.subplots_adjust(hspace=0.3)
    #plt.show()
    plt.savefig('eval_curve.eps', format='eps',bbox_inches='tight')
    
    
    
    
    step_file_name = arglist.plots_dir + arglist.exp_name + '1' + '_train_steps.pkl'
    pkl_file0 = open(step_file_name, 'rb')
    rew1_file_name = arglist.plots_dir + arglist.exp_name + '1' + '_train_formation_err.pkl'
    rew1_file = open(rew1_file_name, 'rb')
    crash1_file_name = arglist.plots_dir + arglist.exp_name + '1' + '_train_crashes.pkl'
    crash1_file = open(crash1_file_name, 'rb')
    rew2_file_name = arglist.plots_dir + arglist.exp_name + '2' + '_train_formation_err.pkl'
    rew2_file = open(rew2_file_name, 'rb')
    crash2_file_name = arglist.plots_dir + arglist.exp_name + '2' + '_train_crashes.pkl'
    crash2_file = open(crash2_file_name, 'rb')
    rew3_file_name = arglist.plots_dir + arglist.exp_name + '3' + '_train_formation_err.pkl'
    rew3_file = open(rew3_file_name, 'rb')
    crash3_file_name = arglist.plots_dir + arglist.exp_name + '3' + '_train_crashes.pkl'
    crash3_file = open(crash3_file_name, 'rb')
    rew4_file_name = arglist.plots_dir + arglist.exp_name + '4' + '_train_formation_err.pkl'
    rew4_file = open(rew4_file_name, 'rb')
    crash4_file_name = arglist.plots_dir + arglist.exp_name + '4' + '_train_crashes.pkl'
    crash4_file = open(crash4_file_name, 'rb')
    steps = pickle.load(pkl_file0)
    
    crash1 = np.array(pickle.load(crash1_file))/10
    crash2 = np.array(pickle.load(crash2_file))/10
    crash3 = np.array(pickle.load(crash3_file))/10
    crash4 = np.array(pickle.load(crash4_file))/10

    rew1 = -np.array(pickle.load(rew1_file))/1000
    rew2 = -np.array(pickle.load(rew2_file))/1000
    rew3 = -np.array(pickle.load(rew3_file))/1000
    rew4 = -np.array(pickle.load(rew4_file))/1000
    x = [i for i in range(6,len(steps))]
    xticks = [6, 10, 15, 20, 25, len(steps)-1]
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    plt.title('Tracking error',fontsize=16)
    plt.plot(x,rew1[6:30], 'x--', label='$APF$ with $d_{risk}$')
    plt.plot(x,rew2[6:30], '+--', label='$APF$ with $d_{stop}$')
    plt.plot(x,rew3[6:30], 'r*--', label='Chao Wang, et al.')
    plt.plot(x,rew4[6:30], 'g-', label='Stream-based')
    #plt.plot(x,rew4[5:], label='our method, $d_{safe,1}=0.7m, d_{safe,2}=0.4m$')
    plt.legend(fontsize=14)
    plt.xlabel('Episode (x1000)', fontsize=16)
    plt.ylabel('Average tracking errors (m/s)', fontsize=16)
    plt.xlim(6,len(steps)-1)
    plt.xticks(xticks)
    plt.subplot(122)
    plt.title('Collision rate',fontsize=16)
    plt.plot(x,crash1[6:30], 'x--', label='$APF$ with $d_{risk}$')
    plt.plot(x,crash2[6:30], '+--', label='$APF$ with $d_{stop}$')
    plt.plot(x,crash3[6:30], 'r*--', label='Chao Wang, et al.')
    plt.plot(x,crash4[6:30], 'g-', label='Stream-based')
    #plt.plot(x,crash4[5:len(steps)], label='our method, $d_{safe,1}=0.7m, d_{safe,2}=0.4m$')
    plt.legend(fontsize=14)
    plt.xlabel('Episode (x1000)', fontsize=16)
    plt.ylabel('Average collision (%/s)', fontsize=16)
    plt.xlim(6,len(steps)-1)
    plt.xticks(xticks)
    #plt.show()
    plt.savefig('train_curve.eps', format='eps',bbox_inches='tight')
    '''
