import neat
import multiprocessing
import os
import numpy as np
import matplotlib.pyplot as plt
from env import EnvBreakoutEasy
import pickle
import datetime

def create_folders():
    folders = ["results", 
               "results/checkpoints", 
               "results/genomes", 
               "results/populations", 
               "results/stats"]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"results/run_{timestamp}"
    os.makedirs(run_folder, exist_ok=True)
    
    return run_folder

def create_env(render_mode=None):
    return EnvBreakoutEasy(render_mode=render_mode)

def eval_genome(genome, config):
    env = create_env(render_mode=None)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    total_fitness = 0.0
    num_episodes = 2
    
    for _ in range(num_episodes):
        observation = env.reset()
        fitness = 0.0
        done = False
        while not done:
            output = net.activate(observation)
            action = int(np.argmax(output))
            observation, reward, done, info = env.step(action)
            fitness += reward
        total_fitness += fitness
    
    return total_fitness / num_episodes

def display_game(genome, config):
    env_vis = create_env(render_mode='human')
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    state = env_vis.reset()
    while not done:
        output = net.activate(state)
        action = int(np.argmax(output))
        state, reward, done, info = env_vis.step(action)
        env_vis.render()
    if hasattr(env_vis, "close"):
        env_vis.close()

def run(config_file):
    run_folder = create_folders()
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(show_species_detail=True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    checkpoint_prefix = os.path.join("results/checkpoints", "neat-checkpoint-")
    checkpoint = neat.Checkpointer(generation_interval=5, 
                                   time_interval_seconds=None, 
                                   filename_prefix=checkpoint_prefix)
    population.add_reporter(checkpoint)

    checkpoint_file = input("Load existing population ? (leave empty to start a new evolution): ")
    if checkpoint_file.strip() != "":
        try:
            if not os.path.exists(checkpoint_file) and os.path.exists(os.path.join("results/checkpoints", checkpoint_file)):
                checkpoint_file = os.path.join("results/checkpoints", checkpoint_file)
                
            print(f"Population loaded from {checkpoint_file}...")
            population = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        except Exception as e:
            print(f"Can't load : {checkpoint_file.strip()} \nError : {e}")
            print("New evolution started...")

    num_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"{num_workers} Workers")

    parallel_evaluator = neat.ParallelEvaluator(num_workers, eval_genome)

    num_generations = int(input("Number of generations: "))
    winner = None
    for gen in range(1, num_generations + 1):
        print(f"==== GEN {gen} ====")
        winner = population.run(parallel_evaluator.evaluate, 1)
        if gen % 5 == 0 and gen >= 10:
            display_game(winner, config, nb_steps=1000)
    
    best_genome_path = os.path.join(run_folder, "best_genome.pkl")
    with open(best_genome_path, "wb") as f:
        pickle.dump(winner, f)
    with open(os.path.join("results/genomes", f"best_genome_{os.path.basename(run_folder)}.pkl"), "wb") as f:
        pickle.dump(winner, f)
    
    population_path = os.path.join(run_folder, "final_population.pkl")
    with open(population_path, "wb") as f:
        pickle.dump(population, f)
    with open(os.path.join("results/populations", f"population_{os.path.basename(run_folder)}.pkl"), "wb") as f:
        pickle.dump(population, f)

    stats_path = os.path.join(run_folder, "stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    with open(os.path.join("results/stats", f"stats_{os.path.basename(run_folder)}.pkl"), "wb") as f:
        pickle.dump(stats, f)

    plot_stats(stats_path, run_folder)

    return winner

def plot_stats(stats_file, output_folder):
    with open(stats_file, "rb") as f:
        stats = pickle.load(f)
    
    best_fitness = np.array(stats.get_fitness_stat(max))
    mean_fitness = np.array(stats.get_fitness_mean())
    top_fitnesses = np.array([
        stats.get_fitness_stat(lambda x: sorted(x, reverse=True)[int(len(x)*0.1)])
        for _ in range(len(best_fitness))
    ])
    
    generations = np.arange(len(best_fitness))

    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness, label="Best fitness")
    plt.plot(generations, mean_fitness, label="Mean fitness")
    plt.plot(generations, top_fitnesses, label="Top 10% fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Population Rewards over Generations")
    plt.legend()
    
    plot_path = os.path.join(output_folder, "fitness_plot.png")
    plt.savefig(plot_path)
    
    plt.savefig(os.path.join("results/stats", f"fitness_plot_{os.path.basename(output_folder)}.png"))
    
    plt.show()

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config")
    best_genome = run(config_path)
