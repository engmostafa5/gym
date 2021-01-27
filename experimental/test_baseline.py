
from gym import wrappers
from time import time # just to have timestamps in the files')
import gym
import pandas as pd
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy

#Candidate Interface with schadow Policies
for N in range(4,8):                                     #M.A to Create population of candidates and evaluate them
    env = gym.make('BipedalWalker-v3', speed_knee=N)
    for Run_M in range(3):                               #M.A two nested for loop lead to square matrix NxM
        model = PPO2(MlpPolicy, env,learning_rate=1e-3, verbose=1)       #M.A to Instantiate the agent (learning_rate=1e-3)
        model.learn(total_timesteps=10)                      #M.A to Train the agent 10 times
        model.save("walker_"+str(N)+"_"+str(Run_M))         #M.A save the trained agent to create date set
        
for N in range(4,8):                                     #M.A to Create population of candidates and evaluate them
    Run=[]
    for Run_M in range(3):                               #M.A two nested for loop lead to square matrix NxM
        model.Load("walker_"+str(N)+"_"+str(Run_M))         #M.A Load the trained agent to create date set 
        mean_reward, variance_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)  #M.A to Evaluate the agent
        candidate = mutate(mean_params)             #M.A Load new policy parameters to agent 
        Run.append({'reward':mean(N)})             #M.A to calculate an store the mean of the episodic rewards in a dictionery
        Run.append({'reward':variance(N)})         #M.A to calculate an store the variance of the episodic rewards in a dictionery
     
    
    
    #obs = env.reset()                                    #M.A The current observation of the environment

        #    candidate = mutate(mean_params)             #M.A Load new policy parameters to agent.
         #   action, _states = model.predict(obs)
          #  obs, reward, dones, info = env.step(action)
           # env.render()                                 #M.A Show the env
        #Run.append({'reward':mean(rewards)})             #M.A to calculate an store the mean of the episodic rewards in a dictionery
        #Run.append({'reward':variance(rewards)})         #M.A to calculate an store the variance of the episodic rewards in a dictionery
#model.save("walker_"+str(knee_speed))                    #M.A to Save the agent parameter from neural network
#vector_row = np.array([])                               #M.A feature vector of each policy
#np.exp(vector_row)                                      #M.A Calculate 2**x for all elements in the array
#log_dir = "stats"                                       #M.A Create log dir
#os.makedirs(log_dir, exist_ok=True)




