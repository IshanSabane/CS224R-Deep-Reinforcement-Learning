"""
Defines a pytorch policy as the agent's actor
"""

import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs224r.infrastructure import pytorch_util as ptu
from cs224r.policies.base_policy import BasePolicy


class MLPPolicySL(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    """
    Defines an MLP for supervised learning which maps observations to actions

    Attributes
    ----------
    logits_na: nn.Sequential
        A neural network that outputs dicrete actions
    mean_net: nn.Sequential
        A neural network that outputs the mean for continuous actions
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions

    Methods
    -------
    get_action:
        Calls the actor update function
    forward:
        Runs a differentiable forwards pass through the network
    update:
        Trains the policy with a supervised learning objective
    """
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

    ##################################

    def save(self, filepath):
        """
        :param filepath: path to save MLP
        """
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        :param obs: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        observation = ptu.from_numpy(observation.astype(np.float32))
        # Sample an action using a gaussian probability distribution to sample the actions
        # instead of getting the first action. 
        
        # DONE return the action that the policy prescribes
        action = self(observation)
        dist = distributions.multivariate_normal.MultivariateNormal(action, torch.diag(torch.exp(self.logstd))) 
        action = dist.sample()

        # sampled_action = torch.normal(action, torch.diag(torch.exp(self.logstd)))
        
        return action.detach().cpu().numpy()




    def forward(self, observation: torch.FloatTensor) -> Any:
        """
        Defines the forward pass of the network

        :param observation: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """


        # DONE: implement the forward pass of the network.
        # You can return anything you want, but you should be able to differentiate
        # through it. For example, you can return a torch.FloatTensor. You can also
        # return more flexible objects, such as a
        # `torch.distributions.Distribution` object. It's up to you!
        
        # print("Forward HERE", observation, type(observation) )
        if self.discrete:
            x = self.logits_na(observation.to(ptu.device))        
        else:
            x = self.mean_net(observation.to(ptu.device))

        return x 
        
        # raise NotImplementedError

    def update(self, observations, actions):
        """
        Updates/trains the policy

        :param observations: observation(s) to query the policy
        :param actions: actions we want the policy to imitate
        :return:
            dict: 'Training Loss': supervised learning loss
        """
        # DONE: update the policy and return the loss
        
        self.optimizer.zero_grad()
        
        pred_action = self(torch.from_numpy(observations).to(ptu.device))    
        dist = distributions.multivariate_normal.MultivariateNormal(pred_action, torch.diag(torch.exp(self.logstd))) 
        
        # pred_action = dist.sample()
        # pred_policy = self.get_action(observations)    
        
        loss = - dist.log_prob(torch.from_numpy(actions).to(ptu.device)).mean()


        # loss = torch.nn.functional.mse_loss(torch.from_numpy(actions).to(ptu.device),
        #                                     pred_policy,
        #                                     reduction='mean')

        loss.backward()
        self.optimizer.step()
        
        
        
        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }

