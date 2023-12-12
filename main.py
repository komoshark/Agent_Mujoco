import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
import pickle
from agents.sac import SAC
from utils.logger import Logger
from utils.replay_memory import ReplayMemory
from utils.model_utils import eval_model
from PIL import Image

def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env_name', default="Hopper-v4")
    parser.add_argument('--policy', default="Gaussian", help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--agent_type', default="SAC", help='Agent Type: SAC, DAGGER, MIX')
    parser.add_argument('--run_type', default="train", help='Run Type: train, eval(Will run the existing models and generate .gif)')
    parser.add_argument('--save_gif', default="True", help='generate .gif')
    parser.add_argument('--eval', type=bool, default=True, help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G', help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G', help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G', help='Temperature parameter')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G', help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N', help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=3001, metavar='N', help='maximum number of steps (default: 1000000)')
    parser.add_argument('--num_episodes', type=int, default=5, metavar='N')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N', help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=5, metavar='N', help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=1000, metavar='N', help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N', help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N', help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true", help='run on CUDA (default: False)')

    args = parser.parse_args()
    return args

def setup_environment(args):
# Environment
    env = gym.make(args.env_name, render_mode="rgb_array")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    return env

def train_sac(args, env, save_model = True):
    memory = ReplayMemory(args.replay_size, args.seed)
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    logger = Logger(args)
    steps_list = []
    alpha_list = []
    episode_reward_list = []
    total_numsteps = 0
    updates = 0
    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state, _ = env.reset()
        while not done:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy
            if len(memory) > args.batch_size:
                for i in range(args.updates_per_step):
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                    updates += 1 
            next_state, reward, done, truncated, info = env.step(action) # Step
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            memory.push(np.array(state), action, np.array(reward), next_state, np.array(mask)) # Append transition to memory
            state = next_state
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            
        alpha_list.append(agent.alpha)
        steps_list.append(total_numsteps)
        episode_reward_list.append(episode_reward)
        log_t = "Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2))
        logger.log_text(log_t)
        print(log_t)

        if i_episode % 30 == 0 and save_model is True:
            logger.store_array_list(alpha_list, 'alpha.pkl')
            logger.store_array_list(episode_reward_list, 'train_reward.pkl')
            logger.store_array_list(steps_list, 'steps.pkl')
            agent.save_checkpoint(args.env_name, suffix = f'{i_episode}')

        if total_numsteps > args.num_steps:
            break

'''
def train_dagger(args, env, save_model = False):
    sac_agent = SAC(env.observation_space.shape[0], env.action_space, args)
    expert_policy = sac_agent.load('pytorch-soft-actor-critic/checkpoints/sac_checkpoint_Hopper-v4_540').policy
    dagger_agent = Dagger(env.observation_space.shape[0], env.action_space, args) 
    total_numsteps = 0
    done = False
    while not done:
        collet_trajectories(args, env, expert_policy)
        for i in range(20):
            daggar_agent.update_parameters(memory, args.batch_size)
        if total_numsteps > args.num_steps:
            break
'''
if __name__ == '__main__':
    args = parse_arguments()
    env = setup_environment(args)
    print(args.agent_type)
    print(args.run_type)
    if args.agent_type == 'SAC':
        if args.run_type == 'train':
            train_sac(args, env)
        elif args.run_type == 'eval':
            sac_agent = SAC(env.observation_space.shape[0], env.action_space, args)
            sac_agent.load_checkpoint(f'checkpoints/{args.env_name}/sac_checkpoint_Hopper-v4_540', evaluate = True)
            eval_model(args, env, sac_agent, True)