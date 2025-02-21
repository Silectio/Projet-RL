import time
import numpy as np
from agent import DQNAgent
from env import EnvBreakout
import optuna

def train_dqn(agent, env, num_episodes=100, target_update_interval=10):
    rewards_history = []
    
    for episode in range(num_episodes):
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

        rewards_history.append(total_reward)
        print(f"Episode: {episode+1}/{num_episodes} - Reward: {total_reward} - LastAvg: {np.mean(rewards_history[-60:]):.2f}")
        print(f"Epsilon: {agent.epsilon:.3f}, Memory: {len(agent.memory.memory)}, loss: {agent.loss:.2f}, avg_loss: {agent.avg_loss:.2f}")
        print('='*50 + '\n')
        
        # Mise à jour de l'epsilon
        agent.update_epsilon()
        
        # Mise à jour peu fréquente du réseau cible
        if (episode + 1) % target_update_interval == 0:
            agent.update_target_network()
            print(f"Episode {episode+1}: Mise à jour du réseau cible effectuée.")

    print("Training completed")
    return rewards_history


def objective(trial):
    env = EnvBreakout(render_mode=None)
    agent = DQNAgent(
        state_size=env.observation_space[0],
        action_size=env.action_space.n,
        lr=trial.suggest_loguniform("lr", 1e-5, 1e-2),
        max_memory=trial.suggest_int("max_memory", 2000, 20000, step=2000),
        epsilon_decay=trial.suggest_uniform("epsilon_decay", 0.990, 0.999)
    )   

    reward_history = train_dqn(agent, env, num_episodes=1000, target_update_interval=trial.suggest_int("target_update_interval", 2, 30))
    # On évalue la moyenne des 100 dernières récompenses
    last_100_rewards = np.mean(reward_history[-100:])
    return -last_100_rewards

if __name__ == "__main__":
    # env = EnvBreakout(render_mode=None)
    # agent = DQNAgent(
    #     state_size=env.observation_space[0],
    #     action_size=env.action_space.n,
    #     epsilon_decay=0.995,
    #     max_memory=10000,
    #     batch_size=64,
    #     lr=1e-3,

    # )
    # agent.model.train()
    # train_dqn(agent, env, num_episodes=1000, target_update_interval=10)
    # time.sleep(10)


    study = optuna.create_study()
    study.optimize(objective, n_trials=20)
    print("Best hyperparams:", study.best_params)
