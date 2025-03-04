import pandas as pd
import seaborn as sns
import os

import matplotlib.pyplot as plt
def graphs(f:str):
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})

    file_path = f'gp_results/{f}'
    data = pd.read_csv(file_path)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Genetic Programming Results Over Generations', fontsize=16)

    ax1.plot(data['Generation'], data['Best'], 'b-', linewidth=2, label='Best Fitness')
    ax1.plot(data['Generation'], data['Average'], 'g-', linewidth=2, label='Average Fitness')
    ax1.set_ylabel('Fitness Value')
    ax1.legend(loc='upper left')
    ax1.set_title('Best and Average Fitness per Generation')

    ax2.plot(data['Generation'], data['Min'], 'r-', linewidth=2, label='Minimum Fitness')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness Value')
    ax2.legend(loc='upper left')
    ax2.set_title('Minimum Fitness per Generation')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    output_dir = 'gp_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(os.path.join(output_dir, 'fitness_evolution_plot.png'), dpi=300)
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.semilogy(data['Generation'], data['Best'], 'b-', linewidth=2, label='Best Fitness')
    plt.semilogy(data['Generation'], data['Average'], 'g-', linewidth=2, label='Average Fitness')
    plt.semilogy(data['Generation'], data['Min'], 'r-', linewidth=2, label='Minimum Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value (log scale)')
    plt.legend(loc='upper left')
    plt.title('Fitness Evolution in Log Scale')
    plt.grid(True, which="both", ls="-")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fitness_evolution_log_plot.png'), dpi=300)
    plt.show()

class Node:
    def __init__(self, label):
        self.label = label
        self.children = []

def parse_expression(s, i=0):
    def skip_spaces(s, i):
        while i < len(s) and s[i].isspace():
            i += 1
        return i
    i = skip_spaces(s, i)
    token = ""
    while i < len(s) and (s[i].isalnum() or s[i] in ['_', '.', '-']):
        token += s[i]
        i += 1
    token = token.strip()
    node = Node(token) if token else None
    i = skip_spaces(s, i)
    if i < len(s) and s[i] == '(':
        i += 1
        while True:
            i = skip_spaces(s, i)
            if i < len(s) and s[i] == ')':
                i += 1
                break
            child, i = parse_expression(s, i)
            node.children.append(child)
            i = skip_spaces(s, i)
            if i < len(s) and s[i] == ',':
                i += 1
            elif i < len(s) and s[i] == ')':
                i += 1
                break
            else:
                break
    return node, i

def build_graph(node, dot, parent_id=None, counter=[0]):
    current_id = f'node{counter[0]}'
    counter[0] += 1
    dot.node(current_id, node.label)
    if parent_id is not None:
        dot.edge(parent_id, current_id)
    for child in node.children:
        build_graph(child, dot, current_id, counter)


def show_solution(sol:str):
    from graphviz import Digraph
    d = Digraph()
    d.node('A', 'if_then_else')
    d.node('B', 'lt(sin(ARG9), ARG3)')
    d.edge('A', 'B')
    d.render('arbre_ast', format='png', view=True)

    # expression_str = open(f'gp_results/{sol}').read()

    # root, _ = parse_expression(expression_str)
    # dot = Digraph(comment='AST')
    # build_graph(root, dot)
    # dot.render('ast', format='png', view=True)


if __name__ == '__main__':
    # graphs('final_stats_20210625_154202.csv')
    show_solution('best_expression_20250303_142022.txt')

    from graphviz import Digraph



