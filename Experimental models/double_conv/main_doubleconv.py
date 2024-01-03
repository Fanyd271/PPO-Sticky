import random
import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from PPO_doubleconv_diff import Agent
import gymnasium as gym
import time
from torch.distributions.categorical import Categorical
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from torch.utils.tensorboard import SummaryWriter


def seed_initialization(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def sample_actions(action_new, action_old, R):
    action_selected = torch.zeros_like(action_new, dtype=torch.int32)
    for i in range(action_new.shape[0]):
        random_number = random.random()
        if random_number < R:
            action_selected[i] = action_new[i]
        else:
            action_selected[i] = action_old[i]
    return action_selected


# PPO Neural Network Model
def wrap_env(env_id, *args, render=False, capture_video=False):
    def inner():
        if render:
            env = gym.make(env_id, render_mode="human")
        else:
            env = gym.make(env_id, render_mode='rgb_array')
        if capture_video:
            if render:
                print("The gym wrapper only allows to either render or record video.")
            else:
                env = gym.wrappers.RecordVideo(env, f"videos/{args[0]}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30) # random initialization
        env = MaxAndSkipEnv(env, skip=4) # frameskipping
        env = EpisodicLifeEnv(env) # mark the end of game
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env) # details about the game
        env = ClipRewardEnv(env) # renormalize the reward for skipped frames 
        env = gym.wrappers.ResizeObservation(env, (84, 84)) # suggested in DQN papers
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4) # help identify the velocity
        return env
    return inner


# Generalized advantage estimation
def GAE(agent, values, rewards, dones, gamma, gae_lambda, notend_game, next_ob):
    num_steps = rewards.shape[0]
    with torch.no_grad():
        advantages =torch.zeros_like(rewards)
        delta = torch.zeros_like(rewards)
        next_value = agent.get_value(next_ob).reshape(1, -1)
        delta[-1] = rewards[-1] + gamma * next_value * notend_game - values[-1]
        advantages[-1] = delta[-1]
        for t in reversed(range(num_steps - 1)):
            delta[t] = rewards[t] + gamma * values[t + 1] * (1 - dones[t + 1]) - values[t]
            advantages[t] = delta[t] + gamma * gae_lambda * (1 - dones[t + 1]) * advantages[t + 1]
        returns = advantages + values
    return advantages, returns


# Training Loop
def train_ppo(agent, device, run_name, total_timesteps, seed, num_steps, num_envs, minibatches, learning_rate, anneal_lr=True, gamma=0.99,
              update_epochs=4, gae_lambda=0.95, clip_eps=0.1, max_grad_norm = 0.5, ent_coef=0.01, vf_coef=0.5, clip_threshold=0.1, record_info=True, early_stop = -1):
    start_time = time.time()
    if record_info:
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            f"gamma: {gamma}|gae_lambda: {gae_lambda}| clip_eps: {clip_eps}|max_grad_norm: {max_grad_norm}|ent_coef: {ent_coef}|vf_coef: {vf_coef}|\
                total_timesteps: {total_timesteps}| num_steps:{num_steps}| num_envs: {learning_rate}| minibatches: {minibatches}| learning_rate: {learning_rate}|\
                clip_threshold: {clip_threshold}| update_epochs: {update_epochs}",
        )
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    envs = agent.envs

    batch_size = int(num_envs * num_steps)
    num_iterations = int(total_timesteps // batch_size)
    train_step = 0
    current_return = -20

    # Initialize the stroage space
    obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_envs)+ envs.single_action_space.shape, dtype=torch.int32).to(device)
    actions_old = torch.zeros((num_steps + 1, num_envs)+ envs.single_action_space.shape, dtype=torch.int32).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    state, _ = envs.reset(seed=seed)
    next_ob = torch.Tensor(state).to(device)
    next_done = torch.zeros(num_envs).to(device)

    # Initialize the actions old buffer
    # Collect the data
    for iteration in range(1, num_iterations + 1):
        if anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations  # learning rate will be decreased
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        actions_old[0] = actions_old[-1] # Determined by the last iteration
        for k in range(0, num_steps):
            train_step = train_step + num_envs
            obs[k] = next_ob
            dones[k] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_ob)
                values[k] = value.flatten()
                actions[k] = action
                logprobs[k] = logprob
            # sample the action
            action = sample_actions(action, actions_old[k].long(), 0.75) # 25% probs to choose the present action
            actions_old[k + 1] = action
            # train
            next_ob, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            rewards[k] = torch.tensor(reward).to(device)
            next_done = np.logical_or(terminations, truncations)
            next_ob, next_done = torch.Tensor(next_ob).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        current_return = info["episode"]["r"]
                        idx = [i for i,x in enumerate(infos["_final_info"]) if x == True ] # find the finished agents
                        actions_old[k + 1][idx] = 0 # this stands for we need to initialize
                        if record_info:
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], train_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], train_step)
        notend_game = 1 - next_done # whether the game ends
        advantages, returns = GAE(agent, values, rewards, dones, gamma, gae_lambda, notend_game, next_ob)
        # train the network
        inds = np.arange(batch_size)
        minibatchsize = batch_size // minibatches

        # flatten the batch to form the right batch size 
        batch_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        batch_logprobs = logprobs.reshape(-1)
        batch_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        batch_advantages = advantages.reshape(-1)
        batch_returns = returns.reshape(-1)
        batch_values = values.reshape(-1)

        for epoch in range(update_epochs):
            np.random.shuffle(inds)
            for start_ind in range(0, batch_size, minibatchsize):
                end_ind = start_ind + minibatchsize
                selected_inds = inds[start_ind:end_ind]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(batch_obs[selected_inds], batch_actions.long()[selected_inds])
                logratio = newlogprob - batch_logprobs[selected_inds]
                ratio = logratio.exp()

                norm_adavantages = batch_advantages[selected_inds]
                norm_adavantages = (norm_adavantages - norm_adavantages.mean())/(norm_adavantages.std() + 1e-8)
                # clip the surrogate loss
                sur_loss1 = -norm_adavantages * ratio
                sur_loss2 = -norm_adavantages * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                sur_loss = torch.max(sur_loss1, sur_loss2).mean()
                # clip the value loss, recommended techiniques in the paper
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - batch_returns[selected_inds]) ** 2
                newvalue_clipped = torch.clamp(newvalue, 
                    batch_values[selected_inds] - clip_threshold, batch_values[selected_inds] + clip_threshold)
                v_loss_clipped = (newvalue_clipped - batch_returns[selected_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
                # entropy loss
                entropy_loss = entropy.mean()

                loss = sur_loss + vf_coef * v_loss - ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()
        if iteration % 5 ==0:
            remain_step = total_timesteps-train_step
            SPS = train_step / (time.time() - start_time)
            print(f"Iter: {iteration}/{num_iterations} | SPS: {int(SPS)} | Episodic return: {int(current_return)} | Remaining time: {int(remain_step/(SPS*3600))}h {int(((remain_step/SPS)%3600)/60)}m")
        if record_info:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], train_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), train_step)
            writer.add_scalar("losses/policy_loss", sur_loss.item(), train_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), train_step)
            writer.add_scalar("charts/SPS", int(train_step / (time.time() - start_time)), train_step)
        if early_stop > 0 and train_step >= early_stop:
            make_dir("model/"+run_name)
            torch.save(agent.actor.state_dict(), "./model/"+run_name+f"/actor_{early_stop}.pth")
            torch.save(agent.critic.state_dict(), "./model/"+run_name+f"/critic_{early_stop}.pth")
            torch.save(agent.network.state_dict(), "./model/"+run_name+f"/network_{early_stop}.pth")
            torch.save(agent.infer_last.state_dict(), "./model/"+run_name+f"/infer_last_{early_stop}.pth")
            print("---------The model early stops----------")
            envs.close()
            return
    envs.close()
    if record_info:
        make_dir("model/"+run_name)
        torch.save(agent.actor.state_dict(), "./model/"+run_name+f"/actor_{early_stop}.pth")
        torch.save(agent.critic.state_dict(), "./model/"+run_name+f"/critic_{early_stop}.pth")
        torch.save(agent.network.state_dict(), "./model/"+run_name+f"/network_{early_stop}.pth")
        torch.save(agent.infer_last.state_dict(), "./model/"+run_name+f"/infer_last_{early_stop}.pth")


def test(device, env_id, path, episodes, render=True, capture_video=False):
    print("--------Start testing-----------")
    env = wrap_env(env_id, path, render=render, capture_video=capture_video)()
    agent = Agent(env).to(device)
    agent.actor.load_state_dict(torch.load("./model/"+path+"/al1.pth"))
    agent.critic.load_state_dict(torch.load("./model/"+path+"/al2.pth"))
    agent.network.load_state_dict(torch.load("./model/"+path+"/network.pth"))
    agent.infer_last.load_state_dict(torch.load("./model/"+path+"/infer_last.pth"))
    for _ in range(episodes):
        state = agent.envs.reset() 
        next_ob = torch.Tensor(state[0]).to(device).unsqueeze(0)
        action_old = torch.zeros((1, ), dtype=torch.int32).to(device)
        done = False

        while not done:
            action, _, _, _ = agent.get_action_and_value(next_ob)
            action = sample_actions(action, action_old, 0.75) # 25% probs to choose the present action
            action_old = action
            next_ob, _, terminations, truncations, info = agent.envs.step(action[0].cpu().numpy())
            next_ob = torch.Tensor(next_ob).to(device).unsqueeze(0)
            done = np.logical_or(terminations, truncations)
            if done == True:
                print("Episodic reward: %10.3f|Episodic length: %10.3f" % (info["episode"]["r"], info["episode"]["l"]))


if __name__ == "__main__":
    seed = 1
    seed_initialization(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_id = "PongNoFrameskip-v4"

    # training parameters
    num_envs = 8
    envs = gym.vector.SyncVectorEnv(
        [wrap_env(env_id) for _ in range(num_envs)]
    )
    agent = Agent(envs).to(device)

    total_timesteps = 50000000 # 50M
    today = datetime.datetime.today()
    run_name = f"{env_id}_{seed}_{today.day}_{datetime.datetime.now().hour}h{datetime.datetime.now().minute}m_{total_timesteps}_doubleconv_diff"
    num_steps = 128
    minibatches = 4
    learning_rate = 2.5e-4
    train_ppo(agent, device, run_name, total_timesteps, seed, num_steps, num_envs, minibatches, learning_rate,
              record_info=True, anneal_lr=True)
    
    # test the model
    # run_name = f"{env_id}_{seed}_{27}_{11}h{37}m_{20000}_vanilla"
    # test(device, env_id, run_name, 1, render=True, capture_video=False)