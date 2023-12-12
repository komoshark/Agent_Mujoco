import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
import torch.nn as nn
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy

class Dagger(object):
    def __init__(self, state_dim, action_space, args):
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.policy = GaussianPolicy(state_dim, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size):
        state_batch, action_batch, _, _, _ = memory.sample(batch_size=batch_size)
        #print('------------------- update_parameters finished -------------------')
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        self.policy_optim.zero_grad()
        action_pred = self.policy(state_batch)
        loss = nn.MSELoss(action_pred, action_batch)
        loss.backward()
        self.policy_optim.step()
        action_pred = self.policy(state_batch)

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/dagger_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            if evaluate:
                self.policy.eval()
            else:
                self.policy.train()