import math
import torch
from utils.logger import Logger
from PIL import Image
import numpy as np

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def eval_model(args, env, agent_, save_gif = False):
    print(f' *************************** Start evaluation ****************************')
    episode_rewards = []
    logger = Logger(args)
    eval_num_episodes = 10
    for i_episode in range(eval_num_episodes):
        num_step = 0
        state, _ = env.reset()
        episode_reward = 0
        done = False
        episode_frames = []
        while not done:
            with torch.no_grad():
                action = agent_.select_action(state, evaluate=True)
            next_state, reward, done, truncated, info = env.step(action)
            num_step += 1
            # Save frames for GIF
            if save_gif and i_episode == eval_num_episodes - 1:
                frame = env.render()
                episode_frames.append(Image.fromarray(frame))    
            episode_reward += reward
            state = next_state
            '''
            if num_step > 100:
                break
            '''
        episode_rewards.append(episode_reward)
        if save_gif and i_episode == eval_num_episodes - 1: # limit the size of gif files
            logger.save_frames_as_gif(episode_frames, filename="evaluation.gif", fps=30)
        print(f'Episode: {i_episode + 1}, Reward: {episode_reward}, Num_step: {num_step}')
    # Save the frames as a GIF
    # Compute and print evaluation metrics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f'Evaluation over {args.num_episodes} episodes: Mean Reward: {mean_reward:.2f}, Std: {std_reward:.2f}')
    return mean_reward, std_reward

def collet_trajectories(args, env, agent_):
    total_numsteps = 0
    for i_episode in range(args.num_episodes):
        episode_reward, episode_steps, done = 0, 0, False
        state, _ = env.reset()
        while not done:
            with torch.no_grad():
                action = agent_.select_action(state, evaluate=True)
            next_state, reward, done, truncated, info = env.step(action)
            mask = 1 if done and not truncated else float(not done)
            memory.push(state, action, reward, next_state, mask) # Append transition to memory
            state = next_state
            total_numsteps += 1
        if total_numsteps > args.num_steps:
            break