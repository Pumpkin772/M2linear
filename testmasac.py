# !/usr/bin/env python 3 torch
# -*- coding: UTF-8 -*-
"""=================================================
@Project -> File   ：Python-Matlab -> train_ddqn
@IDE    ：PyCharm
@Author ：Ymumu
@Date   ：2021/12/3 20:37
@ Des   ：Train MASAC with MTDC
=================================================="""

import matlab.engine
import math
import matplotlib.pyplot as plt
import argparse
import torch
import random
import numpy as np
import time
import datetime
from tensorboardX import SummaryWriter
from copy import deepcopy
import os
import datetime
import pandas as pd
from algo.sac import SAC
from algo.masac import MASAC
from algo.utils import setup_seed
from algo.utils_env import obtain_action, pm_converter, state_reward_CHD, state_reward_DHD, state_reward_CC, \
    output_constraint

rootdir = os.getcwd().replace('\\', '/')
filename = os.path.basename(__file__).split(".")[0]
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = rootdir + '/logs/' + filename + '_'+current_time  # logs for tensorboard


def main(args,epnum):
    setup_seed(20)
    # define the type of env, flag1 and flag2
    train_num = 100
    test_num = 50
    train_env_set = np.random.rand(train_num, 3)
    test_env_set = np.random.rand(test_num, 3)

    save_reward = -200
    # set the control agents
    if args.control_type == 0:  # 集中控制
        agent_num = 1
        state_dim = 12
        action_dim = 16
    else:  # 分布式控制
        agent_num = 2
        state_dim = 5  # p, delta f 3, delta f/t 3
        action_dim = 2  # 2 dimension

    action_bound = 1  # action (-1,1)
    action_scale = 0.99
    # set the environment
    eng = matlab.engine.start_matlab()
    if not args.with_delay:
        env_name = 'M2linear'
    else:
        env_name = 'M4B11_modified_with_delay'

    if not args.linear_control:
        env_name = 'M2linear'
    else:
        env_name = 'M2linear_linear_control'

    eng.load_system(env_name)
    # set the agents
    agents = MASAC(agent_num, state_dim, action_dim, action_scale)

    if not args.uncontrolled:
        print('===============Load the model of sl=================')
        if args.control_type == 0:  # 集中控制
            model_name = 'Centralized_models_v6'
        elif args.control_type == 1:  # 集中计算的H、D
            model_name = 'masac_v0920'
        else:
            model_name = 'masac_v0920'

        ep = 1900
        model_select = 1

        if model_select == 0:
            model_path_pip = rootdir + '/models/' + model_name + '/Best/'
            model_test_name = model_name + '_best'
        elif model_select == 1:
            model_path_pip = rootdir + '/models/' + model_name + '/Final/'
            model_test_name = model_name + '_final'
        else:
            model_path_pip = rootdir + '/models/' + model_name + '/epoch/' + str(epnum) + '/'
            model_test_name = model_name + '_ep' + str(epnum)

        agents.load_model(model_path_pip)

        print('policy net work is loaded!')
        print('===============Model is loaded=================')
    else:
        model_test_name = 'uncontrolled'
    if args.linear_control:
        model_test_name = 'linear_control'
    if args.tensorboard:
        writer = SummaryWriter(logdir)

    # set the train para
    num_training = 0
    batch_size = 256
    auto_entropy = True
    max_episodes = 50  # 训练次数
    counter_env = 0
    counter_eps = 0  # 起始点
    loss_test = []
    data_list = []
    time_list = []
    h_d_list = []
    h_d_time_list = []
    l2norm_list = []
    l2norm_time_list = []
    l2norm_final_list = []
    # begin training
    for ep in range(max_episodes):
        action_step_last = np.zeros(4,)
        counter_step = 0
        t1 = time.time()
        # obtain the initial parameters for environment
        env_index = test_env_set[counter_env]
        if args.with_all_communication:
            env_index[2] = 0.99  # 没有通信线路中断
        counter_env += 1
        if counter_env == train_num:
            counter_env = 0
        flag_type, flag_value, flag_com = env_index[0], env_index[1],  env_index[2]  # random seed for env
        eng.M2Env_init(float(flag_type), float(flag_value), float(flag_com), nargout=0)  # 两个参数为随机种子
        pause_time = 0.2
        if args.linear_control:
            pause_time = 10.2
        stop_time = 10.4
        eng.set_param(env_name+'/Network1/pause_time', 'value', str(pause_time), nargout=0)
        eng.set_param(env_name, 'StopTime', str(10.5), nargout=0)
        eng.set_param(env_name, 'SimulationCommand', 'start', nargout=0)

        states_list, rewards_list, actions_list, dones_list = [], [], [], []
        actions_history = []
        rewards_f_list, rewards_h_list, rewards_d_list = [], [], []
        actions_time_list = []
        done = 0.0
        actions_history.append(np.zeros(4))
        step_max = int(round(10.0/pause_time))
        while True:
            model_status = eng.get_param(env_name, 'SimulationStatus')
            if model_status == 'paused':
                counter_step += 1
                counter_eps += 1
                time_env = np.array(eng.eval('tout')).reshape(-1)
                actions_time_list.append(time_env[-1]+11*ep)
                data = eng.eval('logout.get(\'Sim2Python\').Values.Data')
                # obtain the state and reward
                if args.control_type == 0:  # 集中控制
                    states, rewards, rewards_f, rewards_h, rewards_d = state_reward_CC(data, agent_num, counter_step,
                                                                                       step_max, args, env_index,
                                                                                       actions_history)
                elif args.control_type == 1:  # 集中H、D
                    states, rewards, rewards_f, rewards_h, rewards_d = state_reward_CHD(data, agent_num, counter_step,
                                                                                        step_max, args, env_index,
                                                                                        actions_history)
                    states2, rewards, rewards_f, rewards_h, rewards_d = state_reward_CC(data, agent_num, counter_step,
                                                                                        step_max, args, env_index,
                                                                                        actions_history)
                else:   # 分布式H、D
                    states, rewards, rewards_f, rewards_h, rewards_d = state_reward_DHD(data, agent_num, counter_step,
                                                                                        step_max, args, env_index,action_step_last)
                    states2, rewards, rewards_f, rewards_h, rewards_d = state_reward_CC(data, agent_num, counter_step,
                                                                                        step_max, args, env_index,
                                                                                        action_step_last)
                states_list.append(states)
                rewards_list.append(rewards)
                rewards_f_list.append(rewards_f)
                rewards_h_list.append(rewards_h)
                rewards_d_list.append(rewards_d)
                if args.tensorboard:
                    writer.add_scalar('Reward_EP/rd', np.sum(rewards[0:step_max][:]), global_step=counter_eps)
                    writer.add_scalar('Reward_EP/rd_1', np.sum(rewards_f[0:step_max][:]), global_step=counter_eps)
                    writer.add_scalar('Reward_EP/rd_2', np.sum(rewards_h[0:step_max][:]), global_step=counter_eps)
                    writer.add_scalar('Reward_EP/rd_3', np.sum(rewards_d[0:step_max][:]), global_step=counter_eps)

                # obtain the action
                actions_step = []
                actions_save = []
                for agent_id in range(agent_num):
                    obs = states[agent_id]
                    if not args.uncontrolled:
                        action = agents.model[agent_id].policy_net.get_action(obs, deterministic=True)
                    else:
                        if agent_num == 1:
                            action = np.ones(4)*(-0.0)
                        else:
                            action = np.ones(2)*(-0.0)
                    actions_save.append(action)
                    actions_step.extend(action)

                dones_list.append(done)
                # update the state
                actions_step_num = np.array(actions_step, dtype=float)
                action_step_last = action_step_last * 0 + actions_step_num
                for agent_id in range(agent_num):
                    action_step_last[2*agent_id] = 50*action_step_last[2*agent_id]
                    action_step_last[2 * agent_id+1] = 100*action_step_last[2 * agent_id+1]
                if args.uncontrolled:
                    action_step_last = np.zeros(4, )

                if args.control_type == 0.0:
                    action_step_last = output_constraint(action_step_last)

                    # action_step_last[6] = 0 - action_step_last[0] - action_step_last[2] - action_step_last[4]
                    # action_step_last[7] = 0 - action_step_last[1] - action_step_last[3] - action_step_last[5]
                    # if -100 < action_step_last[6] < 300:
                    #     action_step_last[6] = action_step_last[6]
                    # elif action_step_last[6] <= -100:
                    #     action_step_last[6] = -99
                    # else:
                    #     action_step_last[6] = 299
                    #
                    # if -200 < action_step_last[7] < 600:
                    #     action_step_last[7] = action_step_last[7]
                    # elif action_step_last[7] <= -200:
                    #     action_step_last[7] = -199
                    # else:
                    #     action_step_last[7] = 599

                actions_history.append(np.array(action_step_last))
                actions_list.append(np.array(action_step_last))
                actions_step_str = pm_converter(action_step_last)
                eng.set_param(env_name+'/Python2Sim', 'value', actions_step_str, nargout=0)
                pause_time += 0.2
                print(
                    'EP: {}/{} | Env time : {:.3f} | Running Time: {:.4f}'.format(counter_step, ep, time_env[-1],
                                                                                  time.time() - t1))
                if (pause_time + 0.2) > stop_time:
                    time_ep = time_env + 11*ep
                    time_list.extend(time_ep)
                    data_list.extend(np.array(data))
                    actions_list = np.array(actions_list)
                    h_d_list.extend(np.hstack((actions_list, np.sum(actions_list[:, 0:4:2]/2, axis=1).reshape(-1, 1),
                                               np.sum(actions_list[:, 1:4:2]/2, axis=1).reshape(-1, 1))))
                    h_d_time_list.extend(np.array(actions_time_list))

                    # quantifying the deviation from synchrony
                    l2norm = eng.eval('logout.get(\'L2norm\').Values.Data')
                    l2norm_list.extend(np.array(l2norm))
                    l2norm_time = np.array(eng.eval('tout')).reshape(-1)
                    l2norm_time_list.extend(l2norm_time + 11*ep)
                    l2norm_final_list.append([ep, np.array(l2norm)[-1, -1]])

                    dones_list[-1] = 1.0
                    eng.set_param(env_name, 'SimulationCommand', 'Stop', nargout=0)
                    # update the reply buffer with state, action, reward done
                    ep_rd = np.sum(rewards_list[0:step_max][:])
                    ep_rd_f = np.sum(rewards_f_list[0:step_max][:])
                    ep_rd_h = np.sum(rewards_h_list[0:step_max][:])
                    ep_rd_d = np.sum(rewards_d_list[0:step_max][:])
                    ep_rd_h2 = np.sum(rewards_h_list[step_max - 1][:])
                    ep_rd_d2 = np.sum(rewards_d_list[step_max - 1][:])
                    time_now = datetime.datetime.now().strftime("%H%M%S.%f")
                    loss_test.append([float(time_now), ep, ep_rd, ep_rd_f, ep_rd_h, ep_rd_d, ep_rd_h2, ep_rd_d2])
                    print(
                        'Episode: {}/{} | Episode Reward: {:.4f} ({:.4f}, {:4f}, {:4f})  '
                        '| Running Time: {:.4f}'.format(ep, max_episodes, ep_rd, ep_rd_f, ep_rd_h, ep_rd_d,
                                                        time.time() - t1))
                    reward_ep = np.sum(rewards_list[0:step_max][:])

                    if args.tensorboard:
                        writer.add_scalar('Reward/train_rd', ep_rd, global_step=ep)
                        writer.add_scalar('Reward/train_rd_1', ep_rd_f, global_step=ep)
                        writer.add_scalar('Reward/train_rd_2', ep_rd_h, global_step=ep)
                        writer.add_scalar('Reward/train_rd_3', ep_rd_d, global_step=ep)

                        writer.add_scalar('Reward/train_rd_Hfinal', ep_rd_h2, global_step=ep)
                        writer.add_scalar('Reward/train_rd_Dfinal', ep_rd_d2,  global_step=ep)

                    break # the simulink is stopped at 5s

                eng.set_param(env_name+'/Network1/pause_time', 'value', str(pause_time), nargout=0)
                eng.set_param(env_name, 'SimulationCommand', 'continue', nargout=0)

            elif model_status == 'stopped':  # 仿真有可能会不收敛
                print('仿真异常停止')
                break

    print('===============Save loss in the data.csv=================')
    if args.save_data:
        if args.with_delay:
            model_test_name = model_test_name + '_with_delay'
        else:
            model_test_name = model_test_name
        if args.with_all_communication:
            model_test_name = model_test_name + '_all_communication'

        loss_test = np.array(loss_test)
        pd_rl_loss_test = pd.DataFrame(loss_test, columns=['time', 'step', 'loss', 'loss1', 'loss2', 'loss3', 'loss21',
                                                           'loss31'])
        data_save_path = rootdir + '/test0920/' + model_test_name + '/'
        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path)
        save_name_loss = data_save_path + model_test_name + '_loss_test' + '.csv'
        pd_rl_loss_test.to_csv(save_name_loss, sep=',', header=True, index=False)

        time_test = np.array(time_list).reshape(-1, 1)
        data_test = np.array(data_list)
        data_save = np.hstack((time_test, data_test))
        pd_data_test = pd.DataFrame(data_save)
        save_name_data = data_save_path + model_test_name + '_data_test' + '.csv'
        pd_data_test.to_csv(save_name_data, sep=',', header=True, index=False)

        action_time_test = np.array(h_d_time_list).reshape(-1, 1)
        action_data_test = np.array(h_d_list)
        action_data_save = np.hstack((action_time_test, action_data_test))
        pd_data_test = pd.DataFrame(action_data_save, columns=['time', 'H1', 'D1', 'H2', 'D2',
                                                               'Hsum', 'Dsum'])
        save_name_action = data_save_path + model_test_name + '_action_data_test' + '.csv'
        pd_data_test.to_csv(save_name_action, sep=',', header=True, index=False)

        l2norm_time_test = np.array(l2norm_time_list).reshape(-1, 1)
        l2norm_data_test = np.array(l2norm_list)
        l2norm_data_save = np.hstack((l2norm_time_test, l2norm_data_test))
        pd_data_test = pd.DataFrame(l2norm_data_save)
        save_name_action = data_save_path + model_test_name + '_l2norm_data_test' + '.csv'
        pd_data_test.to_csv(save_name_action, sep=',', header=True, index=False)

        l2norm_final_list = np.array(l2norm_final_list)
        pd_data_test = pd.DataFrame(l2norm_final_list, columns=['step', 'l2norm'])
        save_name_data = data_save_path + model_test_name + '_l2norm_final_test' + '.csv'
        pd_data_test.to_csv(save_name_data, sep=',', header=True, index=False)

    eng.quit()

    # Plot the data
    save_name_data = rootdir + '/test0920/' + model_test_name + '/' + model_test_name + '_data_test' + '.csv'
    data_plot = pd.read_csv(save_name_data)
    time_plot = data_plot.iloc[:, 0]
    power_plot = data_plot.iloc[:, 1:3]
    freq_plot = data_plot.iloc[:, [4, 6]]

    action_name_data = rootdir + '/test0920/' + model_test_name + '/' + model_test_name + '_action_data_test' + '.csv'
    action_plot = pd.read_csv(action_name_data)
    action_time_plot = action_plot.iloc[:, 0]
    action_h_plot = action_plot.iloc[:, [1, 3]]
    action_d_plot = action_plot.iloc[:, [2, 4]]

    plt.figure(figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(4, 1, 1)
    plt.plot(time_plot, power_plot)
    plt.legend(['1', '2'])
    plt.subplot(4, 1, 2)
    plt.plot(time_plot, freq_plot)
    plt.legend(['1', '2'])
    plt.subplot(4, 1, 3)
    plt.plot(action_time_plot, action_h_plot)
    plt.legend(['1', '2'])
    plt.subplot(4, 1, 4)
    plt.plot(action_time_plot, action_d_plot)
    plt.legend(['1', '2'])
    plt.tight_layout()
    plt.savefig(data_save_path + 'figure.png')
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard', default=False, action="store_true")
    parser.add_argument('--save_model', default=False, action="store_true")
    parser.add_argument('--save_data', default=True, action="store_true")
    parser.add_argument('--uncontrolled', default=False, action="store_true")
    parser.add_argument('--linear_control', default=False, action="store_true")
    parser.add_argument('--with_delay', default=False, action="store_true")
    parser.add_argument('--with_all_communication', default=False, action="store_true")


    parser.add_argument('--control_type', default=2.0)
    # type = 0：集中控制
    # type = 1：集中的H、D参数，CHD
    # type = 2：分布式观测H、D参数，DHD

    parser.add_argument('--wF', default=100.0)
    parser.add_argument('--wH', default=1.0)
    parser.add_argument('--wD', default=1.0)
    args = parser.parse_args()
    main(args,0)
    # for ep in range(1000, 1901, 100):
    #     main(args,ep)