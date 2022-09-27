from pickle import NONE
import time
import flappy_bird_gym
from collections import deque
import torch
import random
import os 
from utils import plot, plot_loss, plot_rewards
from model import DQN, DQN_Trainer 

class FlappyAgent:

    def __init__(self,policy_net=DQN(4),target_net=DQN(4),capacity=80000,gamma = 0.8):
        self.capacity = capacity
        self.n_games = 0 
        self.model = policy_net
        self.target_model = target_net
        self.gamma = gamma
        self.update_rate = 2
        self.epsilon = 350
        self.epsilon_decay_factor = 0.3
        self.memory = deque(maxlen=self.capacity)
        self.batch_size = int(self.capacity/100)
        self.test_with_load_model = False
        
        if self.test_with_load_model:
            self.load_pre_existing_model()

        self.initialize_target_weights()
        self.trainer = DQN_Trainer(self.model,self.target_model,0.001,self.gamma)

    def initialize_target_weights(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def load_pre_existing_model(self):
        if os.path.exists(r'C:\Users\85297\Desktop\AI games\strongest_model.pth'):
            self.model.load_state_dict(torch.load('strongest_model.pth')) 
            self.model.eval()

    def memorize_experience(self,state,action,reward,new_state,done):
        self.memory.append((state,action,reward,new_state,done))
        
    def experience_replay(self):

        if len(self.memory) > self.batch_size:
            fragment_sample = random.sample(self.memory,self.batch_size)
        else:
            fragment_sample = self.memory

        state,action,reward,new_state,done = zip(*fragment_sample)
        self.trainer.train_step(state,action,reward,new_state,done)
        
    def train_during_play(self,state,action,reward,new_state,done):
        self.trainer.train_step(state,action,reward,new_state,done)
        
    def get_customized_state(self,environment,given_observation):
        vertical_velocity = environment._game.player_vel_y/10
        rotation_angle = environment._game.player_rot/180
        obs = given_observation 
        cur_obs = obs.tolist().copy()

        state = cur_obs + [vertical_velocity,rotation_angle]
        return state

    def get_info_for_reward(self,environment):
        game = environment._game
        return game.player_y

    def get_action(self,state):
        model_action = [0,0]
        stochastic_threshold = random.randint(0,1000)
        self.decaying_epsilon = self.epsilon - self.epsilon_decay_factor*self.n_games
        if stochastic_threshold < self.decaying_epsilon:
            action = random.randint(0,1)
            model_action[action] = 1
            return [action ,model_action]
        state_t = torch.tensor(state,dtype=torch.float)
        state_n = torch.unsqueeze(state_t, 0)
        prediction = self.model(state_n)
        index = torch.argmax(prediction).item()
        action =  1
        if index == 0:
            action = 0
        model_action[index] = 1
        
        return [action , model_action]

    def model_actions(self,state):
        model_action = [0,0]
        state_t = torch.tensor(state,dtype=torch.float)
        prediction = self.model(state_t)
        index = torch.argmax(prediction).item()
        action =  1
        if index == 0:
            action = 0
        model_action[index] = 1
        return [action , model_action]


    def test_loaded_model(self,environment,observations):
        obs = observations
        while True:
            state_0 = self.get_customized_state(environment,obs)
            final_move = self.model_actions(state_0)[0]
            obs, reward, done, info = environment.step(final_move)
            environment.render()
            time.sleep(1 / 30)
            if done:
                obs = environment.reset()

    def get_screen_state(self):
        pass 

    def pre_process_screen(self):
        pass

if __name__ == '__main__':
    env = flappy_bird_gym.make("FlappyBird-v0")
    birdy = FlappyAgent()
    
    obs = env.reset()
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    episodes_count = 0
    max_update_rate = 8
    total_reward = 0
    cum_rewards = []
    # birdy.test_loaded_model(env,obs)
    while True:
        if birdy.n_games >= 25 and birdy.update_rate < max_update_rate:
            birdy.update_rate+=1

        cur_state = birdy.get_customized_state(env,obs)
        y_pen = birdy.get_info_for_reward(env)
        final_actions = birdy.get_action(cur_state)
        env_action = final_actions[0] 
        model_action = final_actions[1]
        if birdy.trainer.loss:
            plot_loss(birdy.trainer.loss)

        # action = env.action_space.sample() 
        # action_threshold = random.randint(0,20)
        # action = 1 
        # if action_threshold < 19:
        #     action = 0
        obs, reward, done, info = env.step(env_action)
        if not done:
            reward+=2
            if y_pen < 0:
                reward += y_pen/env._screen_size[1]
        else:
            reward-=500*abs(cur_state[1])
            
        new_state = birdy.get_customized_state(env,obs)
        score = info['score']
        birdy.train_during_play(cur_state,model_action,reward,new_state,done)
        birdy.memorize_experience(cur_state,model_action,reward,new_state,done)
        total_reward += reward

        if birdy.n_games % birdy.update_rate == 0:
            birdy.initialize_target_weights()

        env.render()
        time.sleep(1 / 30)  # FPS
        
        # Checking if the player is still alive
        if done:
            cum_rewards.append(total_reward)
            total_reward = 0
            obs = env.reset()
            if score > record:
                record = score
                birdy.model.save_model('strongest_model.pth')
            birdy.n_games +=1
            birdy.experience_replay()
            total_score += score 
            mean_score = total_score/birdy.n_games
            plot_mean_scores.append(mean_score)
            plot_scores.append(score)
            # plot(plot_scores,plot_mean_scores,cum_rewards)
            plot_rewards(cum_rewards)

            
