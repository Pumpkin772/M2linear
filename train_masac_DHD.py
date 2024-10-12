# !/usr/bin/env python 3 torch
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Python-Matlab -> train_ddqn
@IDE    ：PyCharm
@Author ：Ymumu
@Date   ：2021/12/3 20:37
@ Des   ：Train MASAC with MTDC
=================================================='''
import matlab.engine
import math, random
import time
import numpy as np
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
from algo.utils_env import obtain_action, pm_converter, state_reward_DHD


rootdir = os.getcwd().replace('\\', '/')
filename = os.path.basename(__file__).split(".")[0]
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = rootdir+ '/logs/' + filename + '_'+current_time # logs for tensorboard


def main(args):
    setup_seed(20)
    # define the type of env, flag1 and flag2
    train_num = 100
    test_num = 50
    train_env_set =  np.random.rand(train_num, 3)
    test_env_set = np.random.rand(test_num, 3)

    model_name = 'masac_v0920'
    save_reward = -200
    model_path_best = rootdir + '/models/' + model_name + '/Best/'  # to save the actor by reinfoecement learning
    model_path_final= rootdir + '/models/' + model_name + '/Final/'  # to save the actor by reinfoecement learning
    if not os.path.exists(model_path_best):
        os.makedirs(model_path_best)
    if not os.path.exists(model_path_final):
        os.makedirs(model_path_final)
    if args.tensorboard:
        writer = SummaryWriter(logdir)
    # set the control agents
    agent_num = 2
    state_dim = 5 # p, delta f 3, delta f/t 3
    action_dim = 2 # 2 demension
    action_bound = 1 # action (-1,1)
    action_scale = 0.99
    # set the environment
    eng = matlab.engine.start_matlab()
    env_name = 'M2linear'
    eng.load_system(env_name)
    # set the agents
    agents =MASAC(agent_num, state_dim, action_dim, action_scale)
    if args.load_model:
        # load the model
        load_model_name =  'sac_idp_v1'
        load_ep = 200
        load_model_path_best = rootdir + '/models/' + load_model_name + '/Best/'  # to save the actor by reinfoecement learning
        load_model_path_final = rootdir + '/models/' + load_model_name + '/Final/'  # to save the actor by reinfoecement learning
        load_model_path_epoch = rootdir + '/models/' + load_model_name + '/epoch/' + str(load_ep) + '/'
        agents.load_model(load_model_path_final)
        print('policy net work is loaded!')
        print('===============Model is loaded=================')
    # set the train para
    num_training = 0
    batch_size = 256
    auto_entropy = True
    max_episodes = 2000 # 训练次数
    counter_env = 0
    counter_eps = 0
    loss_train = []
    # begin training
    for ep in range(max_episodes):
        action_step_last = np.zeros(action_dim*agent_num,)
        counter_step = 0
        t1 = time.time()
        # obtain the initial parameters for environment
        env_index = train_env_set[counter_env]
        counter_env += 1
        if counter_env == train_num:
            counter_env = 0
        flag_type, flag_value, flag_com = env_index[0], env_index[1],  env_index[2] # random seed for env
        eng.M2Env_init(float(flag_type), float(flag_value), float(flag_com), nargout=0)  # 三个参数为随机种子
        PauseTime = 0.2
        StopTime = 10.4
        eng.set_param(env_name+'/Network1/pause_time', 'value', str(PauseTime), nargout=0)
        eng.set_param(env_name, 'StopTime', str(10.5), nargout=0)
        eng.set_param(env_name, 'SimulationCommand', 'start', nargout=0)

        states_list, rewards_list,actions_list, dones_list = [],[],[],[]
        rewardsF_list, rewardsH_list, rewardsD_list = [],[],[]
        done = 0.0
        step_max= int(round(10.0/PauseTime))
        while True:
            ModelStatus = eng.get_param(env_name, 'SimulationStatus')
            if ModelStatus == 'paused':
                counter_step += 1
                counter_eps += 1
                time_env = np.array(eng.eval('tout')).reshape(-1)
                data = eng.eval('logout.get(\'Sim2Python\').Values.Data')
                # obtain the state and reward
                states, rewards, rewardsF, rewardsH, rewardsD= state_reward_DHD(data, agent_num, counter_step,step_max,args,
                                                                            env_index,action_step_last)
                states_list.append(states)
                rewards_list.append(rewards)
                rewardsF_list.append(rewardsF)
                rewardsH_list.append(rewardsH)
                rewardsD_list.append(rewardsD)
                if args.tensorboard:
                    writer.add_scalar('Reward_EP/rd', np.sum(rewards[0:step_max][:]), global_step=counter_eps)
                    writer.add_scalar('Reward_EP/rd_1', np.sum(rewardsF[0:step_max][:]), global_step=counter_eps)
                    writer.add_scalar('Reward_EP/rd_2', np.sum(rewardsH[0:step_max][:]), global_step=counter_eps)
                    writer.add_scalar('Reward_EP/rd_3', np.sum(rewardsD[0:step_max][:]), global_step=counter_eps)

                # obtain the action
                actions_step = []
                actions_save = []
                for agent_id in range(agent_num):
                    obs = states[agent_id]
                    if args.uncontrol == False:
                        action = agents.model[agent_id].policy_net.get_action(obs, deterministic=False)

                    else:
                        action = np.ones(2)*(0)
                    actions_save.append(action)
                    # act_real = action*math.exp(-counter_step)
                    actions_step.extend(action)

                actions_list.append(actions_save)
                dones_list.append(done)
                # update the state
                actions_step_num = np.array(actions_step, dtype=float)
                action_step_last = action_step_last * 0 + actions_step_num
                for agent_id in range(agent_num):
                    action_step_last[2*agent_id] = 50*action_step_last[2*agent_id]
                    action_step_last[2 * agent_id+1] = 100*action_step_last[2 * agent_id+1]
                # for agent_id in range(agent_num):
                #     action_step_last[2*agent_id] = action_step_last[2*agent_id] \
                #         if action_step_last[2*agent_id] >= -0.49 else -0.49
                #     action_step_last[2 * agent_id + 1] = 0.0
                    # action_step_last[2*agent_id+1] = action_step_last[2*agent_id+1] \
                    #     if action_step_last[2*agent_id+1] >= -0.99 else -0.99

                actions_step_str = pm_converter(action_step_last)
                eng.set_param(env_name+'/Python2Sim', 'value', actions_step_str, nargout=0)
                PauseTime += 0.2
                print(
                    'EP: {}/{} | Env time : {:.3f} | Running Time: {:.4f}'.format(counter_step, ep, time_env[-1],
                                                                                  time.time() - t1))
                if (PauseTime + 0.2) > StopTime:
                    dones_list[-1] = 1.0
                    eng.set_param(env_name, 'SimulationCommand', 'Stop', nargout=0)
                    # update the reply buffer with state, action, reward done
                    if args.uncontrol == False:
                        len1 = len(states_list)
                        rewardsD_list_num = np.array(rewardsD_list)
                        rewardsH_list_num = np.array(rewardsH_list)
                        for i1 in range(step_max):
                            for agent_id in range(agent_num):
                                state = states_list[i1][agent_id]
                                action = actions_list[i1][agent_id]
                                reward = rewards_list[i1+1][agent_id]
                                next_state = states_list[i1+1][agent_id]
                                done = dones_list[i1+1]
                                # # 修改reward list 有关惯性系数的值
                                # interval = 2 # 3 10
                                # if i1>=step_max - 10 and i1<=step_max: # 对于最后的10个惯性系数
                                #     reward = reward + args.wH*np.sum(rewardsH_list_num[i1+2:i1+2+interval,agent_id])
                                #     + args.wD*np.sum(rewardsD_list_num[i1+2:i1+2+interval,agent_id])

                                agents.model[agent_id].replay_buffer.push(state, action, reward, next_state, done)# 10 去掉

                        buffer_length =  [len(agents.model[agent_id].replay_buffer) for agent_id in range(agent_num)]
                        if np.min(buffer_length) > batch_size:
                            for _ in range(50):
                                alpha_loss, q_value_loss1, q_value_loss2, policy_loss = agents.train(batch_size,
                                                                                                     reward_scale=1,
                                                                                                     auto_entropy=auto_entropy,
                                                                                                     target_entropy=-1. * action_dim)
                                if args.tensorboard:
                                    writer.add_scalar('Loss/Alpha_loss', alpha_loss, global_step=num_training)
                                    writer.add_scalar('Loss/Q1_loss', q_value_loss1, global_step=num_training)
                                    writer.add_scalar('Loss/Q2_loss', q_value_loss2, global_step=num_training)
                                    writer.add_scalar('Loss/pi_loss', policy_loss, global_step=num_training)
                                    num_training += 1

                    ep_rd = np.sum(rewards_list[0:step_max][:])
                    ep_rdF = np.sum(rewardsF_list[0:step_max][:])
                    ep_rdH = np.sum(rewardsH_list[0:step_max][:])
                    ep_rdD = np.sum(rewardsD_list[0:step_max][:])
                    ep_rdH2 = np.sum(rewardsH_list[step_max - 1][:])
                    ep_rdD2 = np.sum(rewardsD_list[step_max - 1][:])
                    time_now = datetime.datetime.now().strftime("%H%M%S.%f")
                    loss_train.append([float(time_now), ep, ep_rd, ep_rdF, ep_rdH, ep_rdD, ep_rdH2, ep_rdD2])
                    print(
                        'Episode: {}/{} | Episode Reward: {:.4f} ({:.4f}, {:4f}, {:4f})  | Running Time: {:.4f}'
                            .format(ep, max_episodes, ep_rd, ep_rdF,ep_rdH, ep_rdD, time.time() - t1))
                    reward_ep = np.sum(rewards_list[0:step_max][:])
                    if reward_ep > save_reward and args.save_model:
                        agents.save_model(model_path_best)
                        save_reward = reward_ep
                        print('EP {}:Model saved with Test reward {:4f}'.format(ep, save_reward))


                    if args.tensorboard:
                        writer.add_scalar('Reward/train_rd', ep_rd, global_step=ep)
                        writer.add_scalar('Reward/train_rd_1', ep_rdF, global_step=ep)
                        writer.add_scalar('Reward/train_rd_2', ep_rdH, global_step=ep)
                        writer.add_scalar('Reward/train_rd_3', ep_rdD, global_step=ep)

                        writer.add_scalar('Reward/train_rd_Hfinal', ep_rdH2, global_step=ep)
                        writer.add_scalar('Reward/train_rd_Dfinal', ep_rdD2,  global_step=ep)

                    break # the simulink is stopped at 5s

                eng.set_param(env_name+'/Network1/pause_time', 'value', str(PauseTime), nargout=0)
                eng.set_param(env_name, 'SimulationCommand', 'continue', nargout=0)

            elif ModelStatus == 'stopped':  # 仿真有可能会不收敛
                print('仿真异常停止')
                break

        if ep%100==0 and ep!= 0 and args.save_model:
            model_path_pip_epoch = rootdir + '/models/' + model_name + '/epoch/' + str(ep) + '/'
            if not os.path.exists(model_path_pip_epoch):
                os.makedirs(model_path_pip_epoch)
            agents.save_model(model_path_pip_epoch)
            print('=============The model is saved at epoch {}============='.format(ep))

    if args.save_model:
        agents.save_model(model_path_final)
        print('=============The final model is saved!==========')

    print('===============Save loss in the data.csv=================')
    if args.save_data:
        loss_train = np.array(loss_train)
        pd_rl_loss_train = pd.DataFrame(loss_train, columns=['time', 'step','loss', 'loss1', 'loss2', 'loss3','loss21','loss31'])
        data_save_path = rootdir + '/train/loss/'
        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path)
        save_name = data_save_path + model_name + '_train' + '.csv'
        pd_rl_loss_train.to_csv(save_name, sep=',', header=True, index=False)

    eng.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard', default=True, action="store_true")
    parser.add_argument('--save_model', default=True, action="store_true")
    parser.add_argument('--save_data', default=True, action="store_true")
    parser.add_argument('--uncontrol', default=False, action="store_true")
    parser.add_argument('--load_model', default=False, action="store_true")
    parser.add_argument('--wF', default=100.0)  # 100
    parser.add_argument('--wH', default=1)  # 1
    parser.add_argument('--wD', default=1)  # 1
    args = parser.parse_args()
    main(args)