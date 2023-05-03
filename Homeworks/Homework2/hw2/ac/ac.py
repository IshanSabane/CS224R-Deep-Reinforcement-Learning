import hydra
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class RandomAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


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


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, std=0.1):
        super().__init__()

        self.std = std
        
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs):
        
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * self.std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, num_critics, 
                 hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
        
        self.critics = nn.ModuleList([nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1)) 
            for _ in range(num_critics)])

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        return [critic(h_action) for critic in self.critics]


class PixelACAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, num_critics, critic_target_tau, stddev_clip, use_tb):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.use_tb = use_tb
        self.stddev_clip = stddev_clip

        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             num_critics, hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, num_critics, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        dist = self.actor(obs)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
        return action.cpu().numpy()[0]

    def update(self, replay_iter):
        '''
        This function updates the encoder, critic, target critic and 
        policy parameters respectively.
        
        Args: 
        
        replay_iter:
            An itterable that produces batches of tuples 
            (observation, action, reward, discount, next_observation),
            where:
            observation: array of shape [batch, C, H, W] of stacked 
            image observations
            action: array of shape [batch, action_dim]
            reward: array of shape [batch,]
            discount: array of shape [batch,]
            next_observation: array of shape [batch, C, H, W] of stacked 
            image observations
            
        Returns:
        
        metrics: dictionary of relevant metric to be logged.
        '''
        
        metrics = dict()

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch( batch, self.device)

        ### YOUR CODE HERE ###
        # Part(a)
        enc_obs = self.encoder(self.aug(obs.float()))
        enc_next_obs = self.encoder(self.aug(next_obs.float()))

        # Part(b)
        next_action = self.actor(enc_obs).sample()

        # Part(c)
        target_output = self.critic_target(enc_next_obs,next_action) # All crictic outputs in a list
        
        y_target = reward + discount* torch.minimum(*random.sample(target_output,2))

        print('y_target is ', y_target)


        # Part(d)
        output = self.critic(enc_obs, action)  
        loss = torch.Tensor(sum([(x - y_target.detach())**2 for x in output]).float())

        print(loss, loss.shape)       
        # Part(e)
        loss.backward()
        self.encoder_opt.step()
        self.critic_opt.step()

        # Part(f)
        for i in range(len(self.critic.critics)):
            utils.soft_update_params(self.critic.critics[i], self.critic_target.critics[i], self.critic_target_tau)

        # Final Part 

        sampled_action = self.actor(enc_obs.detach()).sample()
        
        actor_targets =(self.critic(enc_obs.detach(), sampled_action))
        actor_loss = torch.Tensor(-(1/len(actor_targets))* sum(actor_targets)).float()

        actor_loss.backward()
        self.actor_opt.step()

        metrics['actor_loss'] = actor_loss
        metrics['critic_loss']= loss
        #####################
        return metrics
    

    def pretrain(self, replay_iter):
        '''
        This function updates the encoder and policy with end-to-end
        behaviour cloning
        
        Args: 
        
        replay_iter:
            An itterable that produces batches of tuples 
            (observation, action, reward, discount, next_observation),
            where:
            observation: array of shape [batch, C, H, W] of stacked 
            image observations
            action: array of shape [batch, action_dim]
            reward: array of shape [batch,]
            discount: array of shape [batch,]
            next_observation: array of shape [batch, C, H, W] of stacked 
            image observations
            
        Returns:
        
        metrics: dictionary of relevant metric to be logged.
        '''
        
        metrics = dict()


        batch = next(replay_iter)
        obs, action, _, _, _ = utils.to_torch(batch, self.device)
        ### YOUR CODE HERE ###
        # print(obs.shape, action.shape)
        # Augment the observation using the encoder.
        ob_aug = self.aug(obs.float())

        # Pass it to the encoder to reduce its dimension space
        f_theta = self.encoder(ob_aug)
        
        # Actor output: A distribution of the action space 
        actor_out = self.actor(f_theta) 

        # Take the negative of the log probability of the given action from 
        # the replay buffer
        self.actor_opt.zero_grad()
        self.encoder_opt.zero_grad()

        loss = -actor_out.log_prob(action)
        print(loss, loss.shape)



        loss.backward()
        
        self.actor_opt.step()
        self.encoder_opt.step()
        metrics['loss'] = loss
        #####################
        return metrics