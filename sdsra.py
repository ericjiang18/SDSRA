import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy


class SDSRA(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval  # the interval of update
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)   # state num; action num; hidden size num.
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)

        self.predictive_model = PredictiveModel(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.predictive_model_optimizer = Adam(self.predictive_model.parameters(), lr=args.lr)

        
        hard_update(self.critic_target, self.critic)    # copy directly

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        # 1. Skill Initialization
        self.skills = [GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device) for _ in range(args.num_skills)]  # assuming you want multiple Gaussian Policies as skills
        self.relevance_scores = [1.0 for _ in range(args.num_skills)]  # Initialize relevance scores as 1.0 for all skills
        self.skill_optims = [Adam(skill.parameters(), lr=args.lr) for skill in self.skills]



    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)  # mask is done or not

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi  # targetQ = min Q - a*H
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)    # q_val = r + gamma+targetQ
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        
        for idx, skill in enumerate(self.skills):
            predicted_action = self.update_skill(idx, memory, batch_size)  # Here we capture the returned value
            intrinsic_reward = F.mse_loss(predicted_action, action_batch, reduction='none').mean(dim=1, keepdim=True)
            self.update_relevance_score(idx, intrinsic_reward)


        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    
    # 2. Skill Management Methods

    def update_relevance_score(self, skill_idx, intrinsic_reward):
        self.relevance_scores[skill_idx] += intrinsic_reward.mean().item()

    
    def update_skill(self, skill_idx, memory, batch_size):
        state_batch, action_batch, _, _, _ = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        
        predicted_action, _, _ = self.skills[skill_idx].sample(state_batch)

        # Calculate prediction error
        prediction_error = F.mse_loss(predicted_action, action_batch, reduction='none').mean(dim=1, keepdim=True)

        # Compute the entropy of the policy
        policy_entropy = self.skills[skill_idx].get_entropy(state_batch)
        
        # Combine prediction error and entropy for intrinsic reward
        intrinsic_reward = prediction_error + 0*policy_entropy
        
        # Update the skill as you would in an actor-critic method:
        _, log_prob, _ = self.skills[skill_idx].sample(state_batch)
        qf1, qf2 = self.critic(state_batch, predicted_action)
        min_q = torch.min(qf1, qf2)
        
        skill_loss = (log_prob * (log_prob - min_q + intrinsic_reward)).mean()  # Incorporate intrinsic reward

        self.skill_optims[skill_idx].zero_grad()
        skill_loss.backward()
        self.skill_optims[skill_idx].step()
        return predicted_action

    # 3. Modified select_action to use SDSRA
    def select_action(self, state, evaluate=False):
        # Select skill based on softmax selection
        skill_idx = self.select_skill()
        skill = self.skills[skill_idx]
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = skill.sample(state)
        else:
            _, _, action = skill.sample(state)
        return action.detach().cpu().numpy()[0], skill_idx  
    
    def select_skill(self):
        probs = F.softmax(torch.tensor(self.relevance_scores), dim=0)
        return np.random.choice(len(self.skills), p=probs.numpy())

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

    ###
    def train_predictive_model(self, state_batch, action_batch, next_state_batch):
        # Assume state_batch, action_batch, next_state_batch are tensors
        predicted_next_state = self.predictive_model(state_batch, action_batch)
        loss = F.mse_loss(predicted_next_state, next_state_batch)
        self.predictive_model_optimizer.zero_grad()
        loss.backward()
        self.predictive_model_optimizer.step()

class PredictiveModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(PredictiveModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, state_dim)  # predicting the next state, so the output is state_dim

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # Concatenate state and action
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        next_state = self.fc3(x)  # No activation, predicting raw state values
        return next_state

    def predict(self, state, action):
        # This function is just a wrapper around the forward pass to provide a clearer interface
        with torch.no_grad():
            return self.forward(state, action)
