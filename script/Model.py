import numpy as np
import pandas as pd
import copy
from matplotlib import pyplot as plt 
from script.Game import TicTakToe
from script.Agents import RuleBaseAgent,ESAgent
from script.Player import Player

class Evolutionary_Model:
    def __init__(self,max_pop = 1000, parent_percent = 0.2):
        """initilize evolutionary model and first population

        Args:
            max_pop (int, optional): maximum population at each epoch. Defaults to 1000.
            parent_percent (float, optional): the percentage of parent after selection. Defaults to 0.2.
        
        """
        self.max_pop = max_pop
        self.parent_percent = parent_percent

        self.best_agent = None
        self.best_reward = -1000
        self.population = self.initial_population()

    def play_tic_tak_toe(self, max_epoch = 100):
        """play tic tak toe evolutionary and return the best agent you have

        you should follow the persudocode in slides but you allow to some changes as long as it return the best agent you have in the model.

        1. intilize population
        2. do
        3.  fitness         <- evaluation
        3.  parent          <- selection
        4.  new_population  <- evolution
        5. return best_agent

        Returns:
            best_agent (ESagent): _description_
        """

        # Set the number of trials for evaluation of each agent
        num_trials = 10

        # Init the environment
        epsilon = 0.5           # probability of the opponent make a random move, experimenting
        env = TicTakToe(epsilon)

        # Init population
        population = self.initial_population()

        # Enter loop
        for epoch in range(max_epoch):
            
            # Evaluate the fitness of each agent in the population based on their rewards
            agents_rewards = []
            for agent in population:
                agent_reward = self.evaluation(env, agent, num_trials)
                agents_rewards.append(agent_reward)

            # Select best parent_percent to create parent pool based their culmulative rewards
            parent_pool = self.selection(agents_rewards, population)

                

    
    def initial_population(self):
        """initilize first population

        Returns:
            population (ESAgent[]): Array of Agents
        """

        population = []

        for pop in range(self.max_pop):
            parent = ESAgent()
            population.append(parent)

        return population
        
    
    def evaluation(self,env,agent,num_trials):
        """
            evaluate the reward for each agent. feel free to have your own reward function.
        """

        culmative_reward = 0

        for _ in range(num_trials):
            env.reset() # Reset the board 

            # Run the trial 
            while not env.terminate:
                action = agent.make_a_move(env.board)
                env.step(action)

            # Get the reward for the current trial and append to culmul reward
            reward = self.reward_function(env)
            culmative_reward += reward

        return culmative_reward / num_trials
    

    def selection(self,rewards,population):
        """
            select the best fit in the population. feel free to have your own selection.
            Make sure you select parent according to parent_percent
        """
        # Convert the population array to numpy array for ease of apply indices
        pop_arr = np.array(population, dtype=object)

        # Calculate the max parent pool count based on the parent percentage
        max_parent_pool = int(self.max_pop * self.parent_percent)

        # Get the sorted indicies from the rewards and use it to filter out best parent for parent pool
        sorted_reward_idx = np.argsort(rewards)
        sorted_population_by_rewards = pop_arr[sorted_reward_idx]
        parent_pool = sorted_population_by_rewards[-max_parent_pool:]

        return parent_pool



            

    def evolution(self,parents):
        """
            evolute new population from parents. 

            be careful about how to reinforce children. You don't want your children perform same as parents and even worser than parents.

            feel free to have your own evolution. In MLP case, you would like to add some noises to weights and bias.
        """

    def reward_function(self,env):
        """
            reward function for each agent
        """
        if env.winner == Player.PLAYER1:
            return 1    # Win
        elif env.winner == Player.PLAYER2:
            return -1   # Loose
        else:
            return 0    # Draw