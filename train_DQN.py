import time
import numpy as np
from agent import DQNAgent
from env import EnvBreakout
import optuna
import logging

def train_dqn(agent, env, num_episodes=100, target_update_interval=10):
    rewards_history = []
    
    for episode in range(num_episodes):
        start_time = time.time()
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _info = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward

        duration = time.time() - start_time
        rewards_history.append(total_reward)
        avg_reward = np.mean(rewards_history[-60:]) if len(rewards_history) >= 60 else np.mean(rewards_history)
        if (episode + 1) % 20 == 0:
            print("=" * 60)
            logging.info("=" * 60)
            print(f"Episode {episode+1:4d}/{num_episodes} | Reward: {total_reward:8.2f} | Moving Avg (60): {avg_reward:8.2f} | Duration: {duration:6.2f}s")
            logging.info(f"Episode {episode+1:4d}/{num_episodes} | Reward: {total_reward:8.2f} | Moving Avg (60): {avg_reward:8.2f} | Duration: {duration:6.2f}s")
            print(f"Memory Size: {len(agent.memory.memory):4d} | Loss: {agent.loss:6.4f} | Avg Loss: {agent.avg_loss:6.4f}")
            if agent.policy == 'boltzmann':
                print(f"Current Temperature (tau): {agent.tau:6.3f}")
            elif agent.policy == 'epsilon_g':
                print(f"Current Epsilon: {agent.epsilon:6.3f}")
            print("=" * 60 + "\n")

        if agent.policy == 'boltzmann':
            agent.update_tau()
        elif agent.policy == 'epsilon_g':
            agent.update_epsilon()
        
        if (episode + 1) % target_update_interval == 0:
            agent.update_target_network()
            print(f"--> Target network updated at episode {episode+1}")
        
    
    logging.info("Training completed")
    print("Training completed")
    return rewards_history


def objective(trial):
    env = EnvBreakout(render_mode=None)
    decay = trial.suggest_float("decay", 0.9, 0.999, log=True)
    min = trial.suggest_float("min", 0.001, 0.2, log=True)
    agent = DQNAgent(
        state_size=env.observation_space[0],
        action_size=env.action_space.n,
        lr=trial.suggest_float("lr", 1e-6, 1e-1, log=True),
        # batch_size=trial.suggest_int("batch_size", 32, 256, step=32),
        policy=trial.suggest_categorical("policy", ["epsilon_g", "boltzmann"]),
        # max_memory=trial.suggest_int("max_memory", 2000, 30000, step=2000),
        max_memory=10000,
        batch_size=64,
        epsilon_decay=decay,
        tau_decay=decay,
        epsilon_min=min,
        tau_min=min
    )   

    reward_history = train_dqn(agent, env, num_episodes=2000, target_update_interval=40)
    last_100_rewards = np.mean(reward_history[-100:])
    return -last_100_rewards


if __name__ == "__main__":
    logging.basicConfig(filename="training_optuna.log", level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    # env = EnvBreakout(render_mode=None)
    # agent = DQNAgent(
    #     state_size=env.observation_space[0],
    #     action_size=env.action_space.n,
    #     epsilon_decay=0.995,
    #     max_memory=10000,
    #     batch_size=128,
    #     lr=2e-3,
    #     policy='boltzmann'  # Choisir 'epsilon_g' ou 'boltzmann'
    # )
    # agent.model.train()
    # train_dqn(agent, env, num_episodes=1500, target_update_interval=35)
    # time.sleep(10)

    study = optuna.create_study()
    study.optimize(objective, n_trials=30,n_jobs=10)
    print("Best hyperparams:", study.best_params)
    logging.info(f"Best hyperparams: {study.best_params}")
