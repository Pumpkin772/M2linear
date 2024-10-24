# !/usr/bin/env python 3 torch
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：R04_Multi_Houses -> masac
@IDE    ：PyCharm
@Author ：Ymumu
@Date   ：2021/8/14 16:08
@ Des   ：MASAC For HouseHolds
=================================================='''
import numpy as np
from algo.sac import SAC

class MASAC():
    def __init__(self, agent_num,state_dim, action_dim, action_scale):
        self.n_agents = agent_num
        self.action_dim = action_dim
        self.model = [SAC(state_dim, action_dim, action_scale) for i in range(self.n_agents)]

    def train(self, batch_size, reward_scale,auto_entropy, target_entropy):
        alpha_loss_list, q_value_loss1_list, q_value_loss2_list, policy_loss_list = [], [], [], []
        for i1 in range(self.n_agents):
            alpha_loss, q_value_loss1, q_value_loss2, policy_loss = self.model[i1].train(batch_size, reward_scale=1,  auto_entropy=auto_entropy, target_entropy=-1. * self.action_dim)
            alpha_loss_list.append(alpha_loss.cpu().detach())
            q_value_loss1_list.append(q_value_loss1.cpu().detach())
            q_value_loss2_list.append(q_value_loss2.cpu().detach())
            policy_loss_list.append(policy_loss.cpu().detach())

        return np.sum(alpha_loss_list), np.sum(q_value_loss1_list), np.sum(q_value_loss2_list), np.sum(policy_loss_list)

    def save_model(self, path):
        for agent_id in range(self.n_agents):
            path_agent = path + 'agent' + str(agent_id) + '_'
            self.model[agent_id].save_model(path_agent)

    def load_model(self, path):
        for agent_id in range(self.n_agents):
            path_agent = path + 'agent' + str(agent_id) + '_'
            self.model[agent_id].load_model(path_agent)





