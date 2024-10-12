# !/usr/bin/env python 3 torch
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Python_Matlab_SAC -> utils_env
@IDE    ：PyCharm
@Author ：Ymumu
@Date   ：2021/12/5 10:20
@ Des   ：utilis for env converter
=================================================='''
import numpy as np
import math

# for the maddqn obtain action

def obtain_action(num, dim):
    # act is 0,1,2...51*51
    stp = round(200 / dim)
    droop = np.array(range(-99, 99, stp)) / 100
    inertia = np.array(range(-99, 99, stp)) / 100
    # obtain the index for droop
    droop_index = int(num % dim)
    inertia_index = int((num - droop_index) / dim)
    act1 = droop[droop_index]
    act2 = inertia[inertia_index]
    act = np.hstack((act1, act2))
    return act

# the python action to matlab parameters
def pm_converter(actions_step_num):
    actions_step_num = np.hstack((actions_step_num[0:16:2],actions_step_num[1:16:2]))
    act_str = '['
    for ele in actions_step_num:
        str_i1 = format(ele, '.4f')
        act_str = act_str + str_i1 + ' '
    act_str = act_str[:-1]
    act_str = act_str + ']'
    return act_str

# calculate the local reward H、D观测器
def state_reward_DHD(data, agent_no, step, step_max, args, env_index, actions_list):
    disconnect = math.floor(env_index[2] * 17)
    wF, wH, wD = args.wF, args.wH, args.wD # the weights for reward
    time_left = np.max([1-float(step/step_max),0])
    variables = np.array(data)[-1, :]  # dim = 36
    inertia = variables[10:12]
    droop = variables[12:]
    power = variables[0:2]
    freq = [variables[3], variables[5]]
    end_flag_F = 1 if step <= step_max else 0.0
    end_flag_D = 1 if step <= step_max else 0.0
    end_flag_H = 1 if step <= step_max else 0.0      # the last 10 H parameters will be punished
    # end_flag_F = step/step_max if step <= step_max else 0.0
    # end_flag_D = 1.0 if step>=step_max - 10 and step <= step_max else 0.0
    # end_flag_H = 1.0 if step>=step_max - 10 and step <= step_max else 0.0 # the last 10 H parameters will be punished

    states, rewards = [],[]
    rewardsF, rewardsH, rewardsD = [],[],[]
    for agent_id in range(agent_no):
        power = variables[agent_id]  # 0
        deltaF = variables[agent_id * 2 + 2: agent_id * 2 + 4]  # 8,9,10
        deltaFT = variables[agent_id * 2 + 6: agent_id * 2 + 8]  # 32,33,34
        state = np.hstack((power, deltaF, deltaFT))
        states.append(state)
        # if disconnect != 16:
        #     disconnect_T = disconnect % 8 + 1
        #     disconnect_R = math.floor(disconnect/8) + 1
        #     disconnect_node = (disconnect_T + 2*disconnect_R-3)%8
        #     disconnect_node = 8 if disconnect_node == 0.0 else disconnect_node
        #     if agent_id + 1 == disconnect_node:
        #         if disconnect_R == 1:
        #             deltaF = [deltaF[1], deltaF[2]]
        #             deltaFT = [deltaFT[1], deltaFT[2]]
        #         else:
        #             deltaF = [deltaF[0], deltaF[2]]
        #             deltaFT = [deltaFT[0], deltaFT[2]]

        # calculate the reward for each agent
        # reward_F = -1*(freq[agent_id]-np.mean(freq))**2 * end_flag_F  # 0810-1034 a negtive value
        reward_F = -1 * (freq[agent_id] - np.mean(freq)) ** 2 * end_flag_F
        # reward_H = -1*inertia[agent_id]**2 * end_flag_H
        # reward_D = -1*droop[agent_id]**2 * end_flag_D
        reward_H = -1*actions_list[2*agent_id]/200*inertia[agent_id] * end_flag_H
        reward_D = -1*actions_list[2*agent_id+1]/200*droop[agent_id] * end_flag_D
        reward = wF*reward_F + wH*reward_H + wD*reward_D
        rewards.append(reward)
        rewardsF.append(reward_F)
        rewardsH.append(reward_H)
        rewardsD.append(reward_D)

    return states, rewards, rewardsF, rewardsH, rewardsD

# calculate the local reward 集中的H、D参数
def state_reward_CHD(data, agent_no, step, step_max, args, env_index,actions_list):
    disconnect = math.floor(env_index[2] * 4) + 1
    wF, wH, wD = args.wF, args.wH, args.wD # the weights for reward
    time_left = np.max([1-float(step/step_max),0])
    variables = np.array(data)[-1, :]  # dim = 36
    inertia = variables[28:32]
    droop = variables[32:]
    power = variables[0:4]
    freq = [variables[6], variables[9], variables[12], variables[15]]
    end_flag_F = 1 if step <= step_max else 0.0
    end_flag_D = 1 if step <= step_max else 0.0
    end_flag_H = 1 if step <= step_max else 0.0 # the last 10 H parameters will be punished
    # end_flag_F = step/step_max if step <= step_max else 0.0
    # end_flag_D = 1.0 if step>=step_max - 10 and step <= step_max else 0.0
    # end_flag_H = 1.0 if step>=step_max - 10 and step <= step_max else 0.0 # the last 10 H parameters will be punished

    states, rewards = [],[]
    rewardsF, rewardsH, rewardsD = [],[],[]
    for agent_id in range(agent_no):
        power = variables[agent_id]  # 0
        deltaF = variables[agent_id * 3 + 4: agent_id * 3 + 7]  # 4,5,6
        deltaFT = variables[agent_id * 3 + 16: agent_id * 3 + 19]  # 16,17,18
        state = np.hstack((power, deltaF, deltaFT))
        states.append(state)
        if agent_id + 1 == disconnect:
            deltaF = [deltaF[1], deltaF[2]]
            deltaFT = [deltaFT[1], deltaFT[2]]
        elif agent_id == disconnect % 4:
            deltaF = [deltaF[0], deltaF[2]]
            deltaFT = [deltaFT[0], deltaFT[2]]

        # calculate the reward for each agent
        reward_F = -1*np.sum((deltaF-np.mean(deltaF))**2) * end_flag_F  # a negtive value
        reward_H = -1*np.sum(actions_list[-1][0:8:2]/400)**2 * end_flag_H
        reward_D = -1*np.sum(actions_list[-1][1:8:2]/400)**2 * end_flag_D
        reward = wF*reward_F + wH*reward_H + wD*reward_D
        rewards.append(reward)
        rewardsF.append(reward_F)
        rewardsH.append(reward_H)
        rewardsD.append(reward_D)

    return states, rewards, rewardsF, rewardsH, rewardsD

# calculate the local reward 集中控制
def state_reward_CC(data, agent_no, step, step_max, args, env_index,actions_list):
    disconnect = math.floor(env_index[2] * 17)
    wF, wH, wD = args.wF, args.wH, args.wD # the weights for reward
    time_left = np.max([1-float(step/step_max),0])
    variables = np.array(data)[-1, :]  # dim = 36
    inertia = variables[10:12]
    droop = variables[12:]
    power = variables[0:2]
    freq =  [variables[3], variables[5]]
    freqT = [variables[7], variables[9]]
    end_flag_F = 1 if step <= step_max else 0.0
    end_flag_D = 1 if step <= step_max else 0.0
    end_flag_H = 1 if step <= step_max else 0.0  # the last 10 H parameters will be punished
    # end_flag_F = step/step_max if step <= step_max else 0.0
    # end_flag_D = 1.0 if step>=step_max - 10 and step <= step_max else 0.0
    # end_flag_H = 1.0 if step>=step_max - 10 and step <= step_max else 0.0 # the last 10 H parameters will be punished

    states, rewards = [], []
    rewardsF, rewardsH, rewardsD = [], [], []
    state = np.hstack((power, freq, freqT))
    states.append(state)
    reward_F = -1 * np.sum((freq - np.mean(freq)) ** 2) * end_flag_F  # a negtive value
    reward_H = -1*np.sum(actions_list[0:4:2]/200)**2 * end_flag_H
    reward_D = -1*np.sum(actions_list[1:4:2]/200)**2 * end_flag_D
    reward = wF*reward_F + wH*reward_H + wD*reward_D
    rewards.append(reward)
    rewardsF.append(reward_F)
    rewardsH.append(reward_H)
    rewardsD.append(reward_D)

    return states, rewards, rewardsF, rewardsH, rewardsD

def output_constraint(action_step_last):
    action_h_sum = np.sum(action_step_last[0:8:2])
    action_d_sum = np.sum(action_step_last[1:8:2])
    action_step_last[0:8:2] = action_step_last[0:8:2] - action_h_sum / 4.0 * np.ones(4)
    action_step_last[1:8:2] = action_step_last[1:8:2] - action_d_sum / 4.0 * np.ones(4)
    if np.max(action_step_last[0:8:2]) > 298:
        scale = np.max(action_step_last[0:8:2]) / 298.0
        action_step_last[0:8:2] = action_step_last[0:8:2] / scale
    if np.min(action_step_last[0:8:2]) < -98:
        scale = np.min(action_step_last[0:8:2]) / (-98.0)
        action_step_last[0:8:2] = action_step_last[0:8:2] / scale
    if np.max(action_step_last[1:8:2]) > 596.0:
        scale = np.max(action_step_last[1:8:2]) / 596.0
        action_step_last[1:8:2] = action_step_last[1:8:2] / scale
    if np.min(action_step_last[1:8:2]) < -196.0:
        scale = np.min(action_step_last[1:8:2]) / (-196.0)
        action_step_last[1:8:2] = action_step_last[1:8:2] / scale
    return action_step_last
