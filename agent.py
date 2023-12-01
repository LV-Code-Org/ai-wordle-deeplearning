import torch
import random
import numpy as np
from collections import deque
from game import WordleAI, guessRandom, load_wordlist
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
WORD_LIST = load_wordlist()


class Agent:

    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(5, 256, len(WORD_LIST))  # Adjust output size
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        state = game.get_state()
        if state is None:
            # Provide a default state if get_state returns None
            state = [0, 0, 0, 0, 0]
        return state

    def remember(self, state, action, reward, next_state, done) -> None:
        # popleft if MAX_MEMORY is reached
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) -> None:
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(
                self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        action_indices = [WORD_LIST.index(action) for action in actions]
        self.trainer.train_step(states, action_indices, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done) -> None:
        action_index = WORD_LIST.index(action)  # Get the index of the action
        self.trainer.train_step(state, action_index, reward, next_state, done)

    def get_action(self, state) -> str:
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            return guessRandom()  # Random word from the list
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            return WORD_LIST[move]  # Convert index to the corresponding word


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = WordleAI()
    # training loop
    while True:
        # print("starting loop")
        # get old/current state
        state_old = agent.get_state(game)
        # print("got state old")
        # get move
        final_move = agent.get_action(state_old)
        # print("got final move")
        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        # print("got state new")
        # train short memory
        agent.train_short_memory(
            state_old, final_move, reward, state_new, done)
        # print("short mem trained")
        # remember and store in memory
        agent.remember(state_old, final_move, reward, state_new, done)
        # print("remembered")
        if done:
            # train the long memory (aka replay or experience replay memory), plot result
            game.reset()
            agent.n_games += 1
            print(f"n_games={agent.n_games}")
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f"Game: {agent.n_games} | Score: {score} | Record: {record}")

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
