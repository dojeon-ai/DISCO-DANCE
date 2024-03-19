from collections import OrderedDict
from torch import distributions as pyd
from torch.distributions import Independent

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import copy
import common.utils as utils

class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h

class DiagGaussianActor(nn.Module):
    '''
    (s,z) --self.trunk--policy_layers--> mu, log_std
    '''
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim, 
                 action_range, log_std_bounds):
        super().__init__()

        self.action_dim = action_dim
        self.action_range = action_range
        self.log_std_bounds = log_std_bounds
        feature_dim = feature_dim if obs_type == 'pixels' else hidden_dim
        
        self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
        
        policy_layers = []
        policy_layers += [
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True)
        ]
        # add additional hidden layer for pixels
        if obs_type == 'pixels':
            policy_layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
        policy_layers += [nn.Linear(hidden_dim, action_dim * 2)] 

        self.policy = nn.Sequential(*policy_layers)

        self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.trunk(obs)
        mu, log_std = self.policy(h).chunk(2, dim=-1) 

        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + (0.5 * (log_std_max-log_std_min) * (log_std + 1))
        std = log_std.exp()

        if (isinstance(mu, torch.Tensor)) and torch.isnan(mu).any():
            import pdb; pdb.set_trace()
        if (isinstance(std, torch.Tensor)) and torch.isnan(std).any(): 
            import pdb; pdb.set_trace()
        
        base_dist = pyd.Normal(mu, std)
        dist = utils.custom_SquashedNormal(loc=mu, scale=std, dist = base_dist, 
                        action_range=self.action_range, action_dim=self.action_dim)

        base_dist = Independent(base_dist, 1)
        dist = Independent(dist, 1)
        
        return dist, base_dist
    
class Critic(nn.Module):
    '''
    (s,z,a) --self.trunk--q_layers--> q-value
    '''
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        self.obs_type = obs_type

        if obs_type == 'pixels':
            # for pixels actions will be added after trunk
            self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                       nn.LayerNorm(feature_dim), nn.Tanh())
            trunk_dim = feature_dim + action_dim
        else:
            # for states actions come in the beginning
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.Tanh())
            trunk_dim = hidden_dim

        def make_q():
            q_layers = []
            q_layers += [
                nn.Linear(trunk_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
            if obs_type == 'pixels':
                q_layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True)
                ]
            q_layers += [nn.Linear(hidden_dim, 1)]
            return nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        inpt = obs if self.obs_type == 'pixels' else torch.cat([obs, action],
                                                               dim=-1)
        h = self.trunk(inpt)
        h = torch.cat([h, action], dim=-1) if self.obs_type == 'pixels' else h

        q1 = self.Q1(h)
        q2 = self.Q2(h)

        return q1, q2

class SACAgent:
    def __init__(self,
                 name,
                 dtype,
                 maze_type,
                 maze_parent,
                 reward_free,
                 obs_type,
                 obs_shape,
                 action_shape,
                 action_range,
                 device,
                 encoder_lr, 
                 actor_lr,
                 critic_lr,
                 alpha_lr,
                 init_alpha,
                 feature_dim,
                 hidden_dim,
                 critic_target_tau,
                 num_expl_steps,
                 update_every_steps,
                 critic_target_update_frequency,
                 actor_update_frequency,
                 nstep,
                 batch_size,
                 init_critic,
                 use_tb,
                 use_wandb,
                 log_std_bounds,
                 use_auto_alpha,
                 save_dir,
                 maximum_timestep,
                 use_prior=False,
                 only_x = None,
                 meta_dim=0):
        self.dtype = dtype
        self.name = name
        self.maze_type = maze_type
        self.maze_parent = maze_parent
        self.reward_free = reward_free
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.action_dim = action_shape[0]
        self.action_range = action_range
        self.hidden_dim = hidden_dim
        self.encoder_lr = encoder_lr
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.init_alpha = init_alpha
        self.update_every_steps = update_every_steps
        self.critic_target_update_frequency = critic_target_update_frequency
        self.actor_update_frequency = actor_update_frequency
        self.batch_size = batch_size
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.log_std_bounds = log_std_bounds
        self.num_expl_steps = num_expl_steps
        self.init_critic = init_critic
        self.feature_dim = feature_dim
        self.use_auto_alpha = use_auto_alpha
        self.save_dir = save_dir
        self.maximum_timestep = maximum_timestep
        self.use_prior = use_prior
        self.only_x = only_x
        self.guide_skill = None
        # models
        if obs_type == 'pixels':
            self.aug = utils.RandomShiftsAug(pad=4)
            self.encoder = Encoder(obs_shape).to(device)
            self.obs_dim = self.encoder.repr_dim + meta_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = obs_shape[0] + meta_dim  

        # Actor
        self.actor = DiagGaussianActor(obs_type, self.obs_dim, self.action_dim,
                                       feature_dim, hidden_dim, action_range, 
                                       log_std_bounds).to(device)

        self.actor_target = DiagGaussianActor(obs_type, self.obs_dim, self.action_dim,
                                       feature_dim, hidden_dim, action_range, 
                                       log_std_bounds).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic, Critic target
        self.critic = Critic(obs_type, self.obs_dim, self.action_dim,
                             feature_dim, hidden_dim).to(device)
        
        self.critic_target = Critic(obs_type, self.obs_dim, self.action_dim,
                             feature_dim, hidden_dim).to(device)
        
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Alpha
        self.log_alpha = torch.tensor(np.log(init_alpha)).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -self.action_dim 
        
        # Optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        
        if obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(),
                                                lr=encoder_lr)
        else:
            self.encoder_opt = None

        self.train()
        self.critic_target.train()

        if self.dtype == 'float64':
            dtype = torch.float64
        elif self.dtype == 'float32':
            dtype = torch.float32
        self.actor.to(dtype)
        self.actor_target.to(dtype)
        self.critic.to(dtype)
        self.critic_target.to(dtype)
        self.log_alpha.to(dtype)

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    

    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        if self.name == 'disco':
            utils.hard_update_params(other.discriminator, self.discriminator)
        else:
            utils.hard_update_params(other.diayn, self.diayn)
        
        utils.hard_update_params(other.actor, self.actor)
        utils.hard_update_params(other.critic.trunk, self.critic.trunk)
        utils.hard_update_params(other.critic_target.trunk, self.critic_target.trunk)

        self.skill_dim = other.skill_dim
        
    
    def get_meta_specs(self):
        from dm_env import specs 
        return (specs.Array((17,), np.float32, 'skill'),)
        # return tuple()

    def init_meta(self):
        return OrderedDict()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta

    def compute_delta(self, obs, sk, delta_dict):
        # D_delta as None
        return None, delta_dict

    def plot_trajectory(self, env, trajectory, save_dir, step, use_wandb, guide_idx = None, delta_dict = None):

        env.plot_trajectory(trajectory = trajectory, 
                            save_dir = save_dir,
                            step = step,
                            use_wandb = use_wandb, 
                            guide_idx = guide_idx, 
                            delta_dict = delta_dict)

    def plot_guide(self, env, trajectory, save_dir, step, use_wandb, guide_idx = None, terminal_idx = None):

        env.plot_guide(trajectory = trajectory, 
                        save_dir = save_dir,
                        step = step,
                        use_wandb = use_wandb, 
                        guide_idx = guide_idx,
                        terminal_idx = terminal_idx)

    def plot_finetune_trajectory(self, env, trajectory, save_dir, step, use_wandb, goal):

        env.plot_finetune_trajectory(trajectory = trajectory, 
                                    save_dir = save_dir,
                                    step = step,
                                    use_wandb = use_wandb,
                                    goal = goal)

    def plot(self, env, trajectory, save_dir, step, use_wandb):

        if self.guide_skill is None: 
            self.plot_trajectory(env = env,
                                 trajectory = trajectory, 
                                 save_dir = save_dir,
                                 step = step,
                                 use_wandb = use_wandb)
                            
        else:
            guide_idx = self.guide_skill.argmax()
            self.plot_trajectory(env = env, 
                                 trajectory = trajectory, 
                                 save_dir = save_dir,
                                 step = step,
                                 use_wandb = use_wandb,
                                 guide_idx = guide_idx.item())

    def act(self, obs, meta, step, 
            eval_mode, eps=1e-2, finetune=False, sac=False):
        # 1. Obs --encoder--> e(obs) --concat--> [e(obs), skill_index]
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        h = self.encoder(obs) 
        inputs = [h]
        
        if not sac:
            for value in meta.values():
                value = torch.as_tensor(value, device=self.device).unsqueeze(0)
                inputs.append(value)
            inpt = torch.cat(inputs, dim=-1) 
        else:
            inpt = inputs[0]

        # 2. [e(obs), skill_index] --actor--> dist
        dist, _ = self.actor(inpt) 

        if eval_mode:
            action = dist.sample() 
        else:
            action = dist.sample()
            if (not finetune):
                if step < self.num_expl_steps:
                    action.uniform_(-self.action_range, self.action_range)
        assert ((action>self.action_range) | (action<-self.action_range)).sum() == 0, "action range error"
        
        action[action == 1.0] = 1.0 - eps
        action[action == -1.0] = -1.0 + eps
        
        return action.cpu().numpy()[0]

    def sample_action_and_KL(self, obs, guide_obs, guide_skill, eps = 1e-2, min_value = 1.0, max_value = 100):
        # obs: (s, z)
        # guide_obs: (s, z*)
        # guide_skill: (z*)
        # Sample action, subgoals and KL-divergence
        # 1. action from π(.|s,z)
        action_dist, normal_dist = self.actor(obs)
        action = action_dist.rsample()  
        
        action[action == self.action_range] = self.action_range - eps
        action[action == -self.action_range] = -self.action_range + eps
        
        # 2. log π(sampled action | s,z)
        log_prob = action_dist.log_prob(action) 
        if guide_skill is not None: 
            # 3. π(.|s,z*)
            with torch.no_grad():
                if self.use_actor_target:
                    _, prior_normal_dist  = self.actor_target(guide_obs)  
                else:
                    _, prior_normal_dist  = self.actor(guide_obs)  
            # 4. log π(sampled action | s,z*)
            # prior_log_prob = prior_action_dist.log_prob(action)
            # 5. D_kl = π (logπ(.|a,z) - logπ(.|a,z*)) ~ (logπ(.|a,z) - logπ(.|a,z*))
            # (1 sample apprixmation)
            # D_KL = log_prob - prior_log_prob
            D_KL = pyd.kl.kl_divergence(normal_dist, prior_normal_dist)
            D_KL = torch.clamp(D_KL, min_value, max_value)
            
            # Clamping
            x = torch.zeros_like(D_KL)
            D_KL = D_KL + torch.where((D_KL <= self.clamp_kl) & (D_KL >= min_value), -D_KL + torch.tensor(min_value).cuda(), x).detach()
                
        else:
            D_KL = torch.zeros_like(log_prob)
  
        return action, D_KL.unsqueeze(1), log_prob.unsqueeze(1)

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        # Different from DQN
        dist, _ = self.actor(next_obs)
        next_action = dist.rsample()  # reparameterized sampling for backprop
        log_prob = dist.log_prob(next_action).unsqueeze(1)  # continuous R.V -> log_prob>1 OK
        
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - (self.alpha.detach() * log_prob)
        target_Q = reward + (discount * target_V)
        target_Q = target_Q.detach()

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
                        F.mse_loss(current_Q2, target_Q)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        if self.use_tb or self.use_wandb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = current_Q1.mean().item()
            metrics['critic_q2'] = current_Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad()
            self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        dist, _ = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).unsqueeze(1)
        actor_Q1, actor_Q2 = self.critic(obs, action)
        
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
         
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # optimize alpha
        if self.use_auto_alpha:
            self.log_alpha_opt.zero_grad()
            alpha_loss.backward()
            self.log_alpha_opt.step()
        
        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['alpha_loss'] = alpha_loss.item()
            metrics['alpha_value'] = self.alpha.item()

        return metrics

    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)

    def update(self, replay_iter, step):
        raise NotImplementedError()

    def finetune_update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        # Different from DQN
        dist, _ = self.actor(next_obs)
        next_action = dist.rsample()  # reparameterized sampling for backprop
        log_prob = dist.log_prob(next_action).unsqueeze(1)  # continuous R.V -> log_prob>1 OK
        
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - (self.alpha.detach() * log_prob)
        target_Q = reward + (discount * target_V)
        target_Q = target_Q.detach()

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
                        F.mse_loss(current_Q2, target_Q)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        if self.use_tb or self.use_wandb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = current_Q1.mean().item()
            metrics['critic_q2'] = current_Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad()
            self.encoder_opt.step()

        return metrics

    def finetune_update_actor(self, obs, step):
        metrics = dict()

        dist, _ = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).unsqueeze(1)
        actor_Q1, actor_Q2 = self.critic(obs, action)
        
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # optimize alpha
        if self.use_auto_alpha:
            self.log_alpha_opt.zero_grad()
            alpha_loss.backward()
            self.log_alpha_opt.step()
        
        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['alpha_loss'] = alpha_loss.item()
            metrics['alpha_value'] = self.alpha.item()

        return metrics

    def finetune_update(self, replay_iter, env, step, reward_coef, sac=False):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)

        terminal, obs, action, extr_reward, discount, next_obs, xy_p, skill = utils.to_torch(
            batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        next_obs = self.aug_and_encode(next_obs)

        reward = reward_coef * extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        if not sac:
            if not self.update_encoder:
                obs = obs.detach()
                next_obs = next_obs.detach()

            # extend observations with skill
            obs = torch.cat([obs, skill], dim=1)
            next_obs = torch.cat([next_obs, skill], dim=1)

        # update critic
        metrics.update(
            self.finetune_update_critic(obs.detach(), action, reward, discount,
                               next_obs.detach(), step))

        # update actor
        metrics.update(self.finetune_update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics