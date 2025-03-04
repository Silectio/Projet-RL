import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
from env import EnvBreakout
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from multiprocessing import Pool, freeze_support
import sys, os

# Déclaration de constantes globales pour la parallélisation
num_actions = 4
num_features = 128
num_bins = 16

def discretize(state, bins):
    return tuple(np.digitize(state[i], bins[i]) - 1 for i in range(len(state)))

def create_bins(num_bins, num_features):
    bins = np.linspace(0, 255, num_bins + 1)[1:-1]
    return [bins] * num_features

# Fonction utilisée pour initialiser la Q-table ; évite le lambda non picklable
def default_q():
    return np.zeros(num_actions)

# Worker qui exécute local_epochs épisodes sur un processus et renvoie toutes les transitions et les récompenses
def run_worker(epsilon, bins, q_table_copy, num_actions, local_epochs):
    env = EnvBreakout(render_mode=None)
    local_transitions = []
    local_rewards = []
    for _ in range(local_epochs):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            disc_state = discretize(state, bins)
            if np.random.rand() < epsilon:
                action = np.random.randint(num_actions)
            else:
                action = int(np.argmax(q_table_copy[disc_state]))
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            disc_next_state = discretize(next_state, bins)
            local_transitions.append((disc_state, action, reward, disc_next_state))
            state = next_state
        local_rewards.append(total_reward)
    return local_transitions, local_rewards

if __name__ == '__main__':
    freeze_support()  # Nécessaire sous Windows

    # Paramètres d'apprentissage
    alpha = 0.005
    gamma = 0.9995
    epsilon = 1.0
    epsilon_min = 0.075

    # Paramètres de l'environnement et de la Q-table
    bins = create_bins(num_bins, num_features)
    q_table = defaultdict(default_q)

    # Paramètres de parallélisation
    global_episodes = 20000
    local_epochs = 20      # Nombre d'épisodes par worker dans chaque itération
    num_workers = 10
    episodes_per_iteration = local_epochs * num_workers  # 400 épisodes par itération
    iterations = global_episodes // episodes_per_iteration  # Ici 20000 / 400 = 50

    # Calcul du facteur de décroissance par itération pour passer de 1.0 à epsilon_min sur toutes les itérations
    decay_rate = epsilon_min ** (1 / iterations)  # ~0.9494
    print(f"Decay rate per iteration: {decay_rate:.4f}")

    reward_history = []
    total_episodes = 0

    pbar = tqdm(total=global_episodes, desc="Training")
    try:
        for i in range(iterations):
            args = [(epsilon, bins, q_table.copy(), num_actions, local_epochs) for _ in range(num_workers)]
            with Pool(processes=num_workers) as pool:
                results = pool.starmap(run_worker, args)
            # Agrégation des transitions et des récompenses issues de tous les workers
            all_transitions = []
            all_rewards = []
            for transitions, rewards in results:
                all_transitions.extend(transitions)
                all_rewards.extend(rewards)
            # Mise à jour de la Q-table globale
            for (s, a, r, s_next) in all_transitions:
                best_next = np.max(q_table[s_next])
                target = r + gamma * best_next
                error = target - q_table[s][a]
                q_table[s][a] += alpha * error
            epsilon = max(epsilon_min, epsilon * decay_rate)
            reward_history.extend(all_rewards)
            total_episodes += episodes_per_iteration
            avg_reward = np.mean(all_rewards)
            pbar.update(episodes_per_iteration)
            pbar.set_postfix({"Epsilon": f"{epsilon:.3f}", "Mean Reward": f"{avg_reward:.2f}"})
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        pbar.close()

    print(f"Final Q-table size: {len(q_table)}")
    print(f"Total Episodes: {total_episodes}")

    plt.figure(figsize=(12, 6))
    plt.plot(reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.grid(True)
    plt.show()
