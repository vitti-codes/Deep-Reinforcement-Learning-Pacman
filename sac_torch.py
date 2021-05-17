import os
import torch as T
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork
from config import Hyper, Constants
CUDA_LAUNCH_BLOCKING=1

class Agent():
    def __init__(self, input_dims, env, n_actions):
        self.memory = ReplayBuffer(input_dims)
        self.n_actions = n_actions
        self.actor_nn = ActorNetwork(input_dims, n_actions=n_actions, name=Constants.env_id+'_actor', max_action=env.action_space.n)
        self.critic_local_1_nn = CriticNetwork(input_dims, n_actions=n_actions, name=Constants.env_id+'_critic_local_1')
        self.critic_local_2_nn = CriticNetwork(input_dims, n_actions=n_actions, name=Constants.env_id+'_critic_local_2')
        self.critic_target_1_nn = CriticNetwork(input_dims, n_actions=n_actions, name=Constants.env_id+'_critic_target_1')
        self.critic_target_2_nn = CriticNetwork(input_dims, n_actions=n_actions, name=Constants.env_id+'_critic_target_2')

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(Constants.device)
        _, max_probability_action = self.actor_nn.sample_action(state)
        return max_probability_action

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < Hyper.batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample_buffer()

        reward = T.tensor(reward, dtype=T.float).to(Constants.device)
        done = T.tensor(done).to(Constants.device)
        next_state = T.tensor(next_state, dtype=T.float).to(Constants.device)
        state = T.tensor(state, dtype=T.float).to(Constants.device)
        action = T.tensor(action, dtype=T.float).to(Constants.device)

        (action_probabilities, log_action_probabilities), _ = self.actor_nn.sample_action(next_state)
        with T.no_grad():
            action_logits1 = self.critic_target_1_nn(next_state)
            q1_new_policy = T.argmax(action_logits1, dim=1, keepdim=True)
            action_logits2 = self.critic_target_2_nn(next_state)
            q2_new_policy = T.argmax(action_logits2, dim=1, keepdim=True)
            q_value = T.min(q1_new_policy, q2_new_policy)
            min_qf_next_target = action_probabilities * (q_value - Hyper.alpha * log_action_probabilities)
            min_qf_next_target_sum = min_qf_next_target.sum(dim=1).unsqueeze(-1)
            not_done = (1.0 - done*1).unsqueeze(-1)
            next_q_value = reward.unsqueeze(-1) + not_done * Hyper.gamma * (min_qf_next_target_sum)

        action_logits1 = self.critic_local_1_nn(state).gather(1, action.long())
        q_value1 = action_logits1.sum(dim=1).unsqueeze(-1)
        action_logits2 = self.critic_local_2_nn(state).gather(1, action.long())
        q_value2 = action_logits2.sum(dim=1).unsqueeze(-1)
        self.critic_local_1_nn.optimizer.zero_grad()
        self.critic_local_2_nn.optimizer.zero_grad()
        critic_1_loss = F.mse_loss(q_value1, next_q_value)
        critic_2_loss = F.mse_loss(q_value2, next_q_value)
        critic_1_loss.backward()
        critic_2_loss.backward()
        self.critic_local_1_nn.optimizer.step()
        self.critic_local_2_nn.optimizer.step()

        (action_probabilities, log_action_probabilities), _ = self.actor_nn.sample_action(state)

        # -------------------------------------------------------------------------------------------
        # Calculates the loss for the actor. This loss includes the additional entropy term
        # CHANGE0003 Soft state-value where actions are discrete
        self.actor_nn.optimizer.zero_grad()
        action_logits1 = self.critic_target_1_nn(state)
        q1_new_policy = T.argmax(action_logits1, dim=1, keepdim=True)
        action_logits2 = self.critic_target_2_nn(state)
        q2_new_policy = T.argmax(action_logits2, dim=1, keepdim=True)
        q_value = T.min(q1_new_policy, q2_new_policy)
        inside_term = Hyper.alpha * log_action_probabilities - q_value
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        policy_loss.backward(retain_graph=True)
        self.actor_nn.optimizer.step()

        self.update_q_weights()

    def update_q_weights(self):
        local_1_parameters = self.critic_local_1_nn.named_parameters()
        local_2_parameters = self.critic_local_2_nn.named_parameters()
        target_1_parameters = self.critic_target_1_nn.named_parameters()
        target_2_parameters = self.critic_target_2_nn.named_parameters()
		
        self.update_network_parameters_line(target_1_parameters, local_1_parameters, Hyper.tau)
        self.update_network_parameters_line(target_2_parameters, local_2_parameters, Hyper.tau)

    def update_network_parameters_line(self, target_params, local_params, tau):
        for target_param, local_param in zip(target_params, local_params):
            target_param[1].data.copy_(tau*local_param[1].data + (1.0-tau)*target_param[1].data)

    def save_models(self):
        print('.... saving models ....')
        self.actor_nn.save_checkpoint()
        self.critic_local_1_nn.save_checkpoint()
        self.critic_local_2_nn.save_checkpoint()
        self.critic_target_1_nn.save_checkpoint()
        self.critic_target_2_nn.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor_nn.load_checkpoint()
        self.critic_local_1_nn.load_checkpoint()
        self.critic_local_2_nn.load_checkpoint()
        self.critic_target_1_nn.load_checkpoint()
        self.critic_target_2_nn.load_checkpoint()

    
