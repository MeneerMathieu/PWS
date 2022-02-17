import logging

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')

LEARNING_RATE = 1
DISCOUNT = 1
RUNS = 100000
RENDER_EVERY = round(RUNS / 10)
UPDATE_EVERY = round(RUNS / 100)

epsilon = 1
MINIMAL_EPSILON = 0.01
START_EPSILON_DECAY = 1
END_EPSILON_DECAY = RUNS * 0.9
epsilon_decay_value = epsilon / (END_EPSILON_DECAY - START_EPSILON_DECAY)

NUM_BINS = 6


def create_bins():
    bins = [
        np.linspace(-4.8, 4.8, NUM_BINS),
        np.linspace(-5, 5, NUM_BINS),
        np.linspace(-0.418, 0.418, NUM_BINS),
        np.linspace(-5, 5, NUM_BINS)
    ]
    return bins


def get_observation_space_size():
    observationspace_size = len(env.observation_space.high)
    return observationspace_size


def create_q_table():
    q_table = np.random.uniform(low=-2, high=0, size=([NUM_BINS] * observation_space_size + [env.action_space.n]))
    return q_table


def get_discrete_state(state, bins, observation_space_size):
    state_index = []
    for i in range(observation_space_size):
        state_index.append(np.digitize(state[i], bins[i]))
    return tuple(state_index)


bins = create_bins()
observation_space_size = get_observation_space_size()
q_table = create_q_table()


previous_score = []
metrics = {'ep': [], 'avg': [], 'min': [], 'max': []}


for run in range(RUNS):
    discrete_state = get_discrete_state(env.reset(), bins, observation_space_size)
    done = False
    count = 0

    while not done:
        if run % RENDER_EVERY == 0:
            env.render()

        count += 1
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state, bins, observation_space_size)

        max_future_q = np.max(q_table[new_discrete_state])
        current_q = q_table[discrete_state + (action, )]

        if done and count < 150:
            reward = -400
        new_q = current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q - current_q)
        q_table[discrete_state + (action, )] = new_q

        discrete_state = new_discrete_state

    previous_score.append(count)

    if epsilon > MINIMAL_EPSILON:
        if END_EPSILON_DECAY >= run >= START_EPSILON_DECAY:
            epsilon -= epsilon_decay_value
            LEARNING_RATE = epsilon

    if run % UPDATE_EVERY == 0:
        latest_runs = previous_score[-UPDATE_EVERY:]
        average_score = sum(latest_runs)  / len(latest_runs)
        metrics['ep'].append(run)
        metrics['avg'].append(average_score)
        metrics['min'].append(min(latest_runs))
        metrics['max'].append(max(latest_runs))
        print("Run:", run, "Average:", average_score, "Min:", min(latest_runs), "Max:", max(latest_runs))


env.close()

# Plot graph
plt.plot(metrics['ep'], metrics['avg'], label="Gemiddelde rewards")
plt.plot(metrics['ep'], metrics['min'], label="Minimale rewards")
plt.plot(metrics['ep'], metrics['max'], label="Maximale rewards")
plt.legend(loc=4)
plt.show()
