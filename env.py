import gym



#Wrapper for the gym environment
class EnvBreakout:
    def __init__(self, render_mode='human'):
        self.env = gym.make('Breakout-v0', render_mode=render_mode)
        self.env.reset()
        observation_ram = self.env.ale.getRAM()
        self.observation_space = observation_ram.shape
        self.action_space = self.env.action_space

    def reset(self):
        self.env.reset()
        return self.RAM_Obs()
    
    def step(self, action):
        _, reward, done, terminated, info  = self.env.step(action)
        observation = self.RAM_Obs()
        return observation, reward, done or terminated, info 
        #(Only 4 outputs as it is more common)
    
    def RAM_Obs(self):
        observation_ram = self.env.ale.getRAM()
        return observation_ram
    
    def render(self):
        self.env.render()



if __name__ == '__main__':
    import time
    env = EnvBreakout()
    print(env.observation_space)
    print(env.action_space)
    print(env.reset())
    print(env.step(1))
    time.sleep(10)