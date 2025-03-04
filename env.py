import gym
import cv2
import random
import time
import numpy as np
from gym import spaces
# Wrapper for the gym environment
class EnvBreakout:
    def __init__(self, render_mode='human'):
        self.env = gym.make('Breakout-v4', render_mode=render_mode)
        self.env.reset()
        observation_ram = self.env.ale.getRAM()
        self.observation_space = observation_ram.shape
        self.action_space = self.env.action_space

    def reset(self):
        self.env.reset()
        return self.RAM_Obs()
    
    def step(self, action):
        _, reward, done, terminated, info = self.env.step(action)
        observation = self.RAM_Obs()
        return observation, reward, done or terminated, info 
        # (Only 4 outputs as it is more common)
    
    def RAM_Obs(self):
        observation_ram = self.env.ale.getRAM()
        return observation_ram
    
    def Pixel_Obs(self):
        # Retourne l'image pixel courante (RGB) via ALE
        return self.env.ale.getScreenRGB()
    
    def render(self):
        return self.env.render()
# Wrapper for the gym environment


class EnvBreakoutEasy:
    def __init__(self, render_mode='human'):
        self.env = gym.make('Breakout-v4', render_mode=render_mode)
        self.env.reset()
        observation_ram = self.RAM_Obs()
        self.observation_space = observation_ram.shape
        # On limite l'espace d'action à 3 : [NOOP, RIGHT, LEFT]
        self.action_space = spaces.Discrete(3)
        # Mapping : agent 0 -> NOOP (0), agent 1 -> RIGHT (2), agent 2 -> LEFT (3)
        self._action_map = {0: 0, 1: 2, 2: 3}
        self.lives = self.env.unwrapped.ale.lives()
        self.waiting_for_fire = False
        self.wait_frames = 0

    def reset(self):
        self.env.reset()
        self.lives = self.env.unwrapped.ale.lives()
        self.waiting_for_fire = True
        self.wait_frames = 3
        return self.RAM_Obs()

    def step(self, action):
        if self.waiting_for_fire:
            if self.wait_frames < 3:
                actual_action = 0  # NOOP pendant l'attente
                self.wait_frames += 1
            else:
                actual_action = 1  # FIRE pour relancer la balle
                self.waiting_for_fire = False
                self.wait_frames = 0
            _, reward, done, terminated, info = self.env.step(actual_action)
            obs = self.RAM_Obs()
            self.lives = self.env.unwrapped.ale.lives()
            return obs, reward, done or terminated, info
        else:
            actual_action = self._action_map[action]
            _, reward, done, terminated, info = self.env.step(actual_action)
            obs = self.RAM_Obs()
            current_lives = self.env.unwrapped.ale.lives()
            if current_lives < self.lives and current_lives > 0:
                self.waiting_for_fire = True
                self.wait_frames = 0
            self.lives = current_lives
            return obs, reward, done or terminated, info

    def RAM_Obs(self):
        observation_ram = self.env.ale.getRAM()
        # On ne garde que certaines colonnes d'intérêt
        observation_ramF = observation_ram[[70, 71, 72, 74, 75, 90, 94, 95, 99, 101, 103, 105, 119]]
        return observation_ramF

    def Pixel_Obs(self):
        return self.env.ale.getScreenRGB()

    def render(self):
        return self.env.render()


if __name__ == '__main__':
    env = EnvBreakoutEasy(render_mode='human')
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    
    state = env.reset()
    print("Initial RAM observation:", state)
    time.sleep(1)
    
    while True:
        action = random.randint(0, env.action_space.n - 1)
        state, reward, done, info = env.step(action)
        
        env.render()
        
        frame = env.Pixel_Obs()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Pixel Observation", frame_bgr)
        
        # Display RAM observation as a grayscale image
        ram_obs = state
        ram_obs_image = np.tile(ram_obs, (10, 1))  # Repeat the RAM values to make it visible
        cv2.imshow("RAM Observation", ram_obs_image)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
        if done:
            env.reset()
    
    cv2.destroyAllWindows()
    env.env.close()
