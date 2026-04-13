import sys
import time
import pickle
from opcode import hasexc

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from vis_gym import *

BOLD = '\033[1m'  # ANSI escape sequence for bold text
RESET = '\033[0m' # ANSI escape sequence to reset text formatting

train_flag = 'train' in sys.argv
gui_flag = 'gui' in sys.argv

setup(GUI=gui_flag)
env = game # Gym environment already initialized within vis_gym.py

#env.render() # Uncomment to print game state info

def hash(obs):
    '''
    Compute a unique compact integer ID representing the given observation.

    Encoding scheme:
      - Observation fields:
          * player_health: integer in {0, 1, 2}
          * window: a 3×3 grid of cells, indexed by (dx, dy) with dx, dy ∈ {-1, 0, 1}
          * guard_in_cell: optional identifier of a guard in the player’s cell (e.g. 'G1', 'G2', ...)

      - Each cell contributes a single digit (0–8) to a base-9 number:
          * If the cell is out of bounds → code = 8
          * Otherwise:
                tile_type = 
                    0 → empty
                    1 → trap
                    2 → heal
                    3 → goal
                has_guard = 1 if one or more guards present, else 0
                cell_value = has_guard * 4 + tile_type  # ranges from 0 to 7

        The 9 cell_values (row-major order: top-left → bottom-right) form a 9-digit base-9 integer `window_hash`.

      - The final state_id packs:
            * window_hash  → fine-grained local state
            * guard_index  → identity of guard in player’s cell (0 if none, 1–4 otherwise)
            * player_health → coarse health component

        Specifically:
            WINDOW_SPACE = 9 ** 9
            GUARD_SPACE  = WINDOW_SPACE       # for guard_index (0–4)
            HEALTH_SPACE = GUARD_SPACE * 5    # for health (0–2)

            state_id = (player_health * HEALTH_SPACE) 
                     + (guard_index * GUARD_SPACE) 
                     + window_hash

    Returns:
        int: A unique, compact integer ID suitable for tabular RL (e.g. as a Q-table key).
    '''
    health = int(obs.get('player_health', 0))
    window = obs.get('window', {})

    # Build cell values in a stable order: dx -1..1 (rows), dy -1..1 (cols)
    cell_values = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            cell = window.get((dx, dy))
            if cell is None or not cell.get('in_bounds', False):
                cell_values.append(8)
                continue

            # Determine tile type
            if cell.get('is_trap'):
                tile_type = 1
            elif cell.get('is_heal'):
                tile_type = 2
            elif cell.get('is_goal'):
                tile_type = 3
            else:
                tile_type = 0

            has_guard = 1 if cell.get('guards') else 0
            cell_value = has_guard * 4 + tile_type
            cell_values.append(cell_value)

    # Pack into base-9 integer
    window_hash = 0
    base = 1
    for v in cell_values:
        window_hash += v * base
        base *= 9

    # Include guard identity when player is in the center cell.
    # guard_in_cell is a convenience field set by the environment (e.g. 'G1' or None).
    guard_in_cell = obs.get('guard_in_cell')
    if guard_in_cell:
        # map 'G1' -> 1, 'G2' -> 2, etc.
        try:
            guard_index = int(str(guard_in_cell)[-1])
        except Exception:
            guard_index = 0
    else:
        guard_index = 0

    # window_hash uses 9^9 space; reserve an extra multiplier for guard identity (0..4)
    WINDOW_SPACE = 9 ** 9
    GUARD_SPACE = WINDOW_SPACE  # one slot per guard id
    HEALTH_SPACE = GUARD_SPACE * 5  # 5 possible guard_id values (0 = none, 1-4 = guards)

    state_id = int(health) * HEALTH_SPACE + int(guard_index) * GUARD_SPACE + window_hash
    return state_id

'''
Complete the function below to do the following:

        1. Run a specified number of episodes of the game (argument num_episodes). An episode refers to starting in some initial
             configuration and taking actions until a terminal state is reached.
        2. Maintain and update Q-values for each state-action pair encountered by the agent in a dictionary (Q-table).
        3. Use epsilon-greedy action selection when choosing actions (explore vs exploit).
        4. Update Q-values using the standard Q-learning update rule.

Important notes about the current environment and state representation

        - The environment is partially observable: observations returned by env.get_observation() include a centered 3x3
            "window" around the player plus the player's health. Each observation is a dict with these relevant keys:
                    - 'player_position': (x, y)
                    - 'player_health': integer (0=Critical, 1=Injured, 2=Full)
                    - 'window': a dict keyed by (dx,dy) offsets in {-1,0,1} x {-1,0,1}. Each entry contains:
                                { 'guards': list or None, 'is_trap': bool, 'is_heal': bool, 'is_goal': bool, 'in_bounds': bool }
                    - 'at_trap', 'at_heal', 'at_goal', and 'guard_in_cell' are convenience fields for the center cell.

        - To make a compact and consistent state hash for tabular Q-learning, encode the 3x3 window plus player health into a single integer.
            use the provided hash(obs) function above. Note that the player position is not included in the hash, as it is not needed for local decision-making.

        - Your Q-table should be a dict mapping state_id -> np.array of length env.action_space.n. Initialize arrays to zeros
            when you first encounter a state.

        - The actions available in this environment now include movement, combat, healing and waiting. The action indices are:
                    0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: FIGHT, 5: HIDE, 6: HEAL, 7: WAIT

        - Remember to call obs, reward, done, info = env.reset() at the start of each episode.

        - Use a learning-rate schedule per (s,a) pair, i.e. eta = 1/(1 + N(s,a)) where N(s,a) is the
            number of updates applied to that pair so far.

Finally, return the dictionary containing the Q-values (called Q_table).

'''

def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=0.999):
    """
    Run Q-learning algorithm for a specified number of episodes.

    Parameters:
    - num_episodes (int): Number of episodes to run.
    - gamma (float): Discount factor.
    - epsilon (float): Exploration rate.
    - decay_rate (float): Rate at which epsilon decays. Epsilon should be decayed as epsilon = epsilon * decay_rate after each episode.

    Returns:
    - Q_table (dict): Dictionary containing the Q-values for each state-action pair.
    """
    Q_table = {}
    Q_seen = {}

    episode_rewards = []

    for episode in tqdm(range(num_episodes)):
        # Setup
        obs, reward, done, info = env.reset()
        new_state = hash(obs)
        if new_state not in Q_table:
            Q_table[new_state] = np.zeros(env.action_space.n)
            Q_seen[new_state] = np.zeros(env.action_space.n)
        episode_reward = 0

        # Run episode
        while not done:
            state = new_state

            # Select action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.random.choice(np.where(Q_table[state] == Q_table[state].max())[0])

            obs, reward, done, info = env.step(action)
            new_state = hash(obs)
            episode_reward += reward

            # Update Q-values
            Q_seen[state][action] += 1
            if new_state not in Q_table:
                Q_table[new_state] = np.zeros(env.action_space.n)
                Q_seen[new_state] = np.zeros(env.action_space.n)
            Q_table[state][action] += (reward + gamma * Q_table[new_state].max() - Q_table[state][action]) / Q_seen[state][action]


        episode_rewards.append(episode_reward)
        epsilon *= decay_rate


    # Make plot
    plt.plot(episode_rewards, 'o', label = "Episode Rewards")

    averages = []
    for i in range(num_episodes // 1000):
        averages.append(np.average(episode_rewards[1000*i:1000*(i+1)]))
    plt.plot(range(500, num_episodes, 1000), averages, 'r-', label = "Average over 1000 episodes")

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('episode_rewards.png')

    # Analyze Q-table
    CENTER_CELL = 9 ** 4
    GUARD_SPACE = 9 ** 9
    HEALTH_SPACE = GUARD_SPACE * 5

    Q_totals = np.zeros((5, env.action_space.n))
    Q_weights = np.zeros((5, env.action_space.n))
    for state in Q_table:
        cell = state // CENTER_CELL % 9

        # Heal cell
        if cell == 2:
            Q_totals[0] += Q_table[state]
            Q_weights[0] += Q_seen[state]

        # Guards
        guard = state % HEALTH_SPACE // GUARD_SPACE
        if guard != 0:
            Q_totals[guard] += Q_table[state]
            Q_weights[guard] += Q_seen[state]
    np.savetxt('Q_summary.csv', np.round(Q_totals / Q_weights, 2), delimiter=',', fmt="%.2f")

    return Q_table

'''
Specify number of episodes and decay rate for training and evaluation.
'''

num_episodes = 1000000
decay_rate = 0.999995

'''
Run training if train_flag is set; otherwise, run evaluation using saved Q-table.
'''

if train_flag:
    Q_table = Q_learning(num_episodes=num_episodes, gamma=0.9, epsilon=1, decay_rate=decay_rate) # Run Q-learning

    # Save the Q-table dict to a file
    with open('Q_table.pickle', 'wb') as handle:
        pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)


'''
Evaluation mode: play episodes using the saved Q-table. Useful for debugging/visualization.
Based on autograder logic used to execute actions using uploaded Q-tables.
'''

def softmax(x, temp=1.0):
    e_x = np.exp((x - np.max(x)) / temp)
    return e_x / e_x.sum(axis=0)

if not train_flag:

    rewards = []
    episode_lengths = []
    num_actions = 0
    unknown_states = set()
    unknown_actions = 0

    filename = 'Q_table.pickle'
    input(f"\n{BOLD}Currently loading Q-table from "+filename+f"{RESET}.  \n\nPress Enter to confirm, or Ctrl+C to cancel and load a different Q-table file.\n(set num_episodes and decay_rate in Q_learning.py).")
    Q_table = np.load(filename, allow_pickle=True)

    for episode in tqdm(range(10000)):
        obs, reward, done, info = env.reset()
        total_reward = 0
        episode_length = 0

        while not done:
            state = hash(obs)
            try:
                action = np.random.choice(env.action_space.n, p=softmax(Q_table[state]))  # Select action using softmax over Q-values
            except KeyError:
                action = env.action_space.sample()  # Fallback to random action if state not in Q-table
                unknown_states.add(state)
                unknown_actions += 1

            obs, reward, done, info = env.step(action)
            episode_length += 1

            total_reward += reward
            if gui_flag:
                refresh(obs, reward, done, info, delay=.1)  # Update the game screen [GUI only]

        num_actions += episode_length

        #print("Total reward:", total_reward)
        rewards.append(total_reward)
        episode_lengths.append(episode_length)
    avg_reward = sum(rewards)/len(rewards)
    avg_episode_length = sum(episode_lengths)/len(episode_lengths)

    print(f"Average episode length: {avg_episode_length}")
    print(f"Average episode reward: {avg_reward}")
    print(f"States in Q-table: {len(Q_table)}")
    print(f"Unknown states encountered: {len(unknown_states)}")
    print(f"Actions from unknown states: {unknown_actions*100/num_actions}%")
