import neat
import gym
import pickle
import time
import numpy as np
from gym.wrappers import RecordVideo, RecordEpisodeStatistics

NUMEPISODIOS = 1
NUMSTEPS = 300


def translate_output(output):
    output_range = (-2, 2)
    for i in range(len(output)):
        if (output[i] >= output_range[1]):
            output[i] = output_range[1]
        elif (output[i] <= output_range[0]):
            output[i] = output_range[0]
        else:
            output[i] = output[i] * 2
    return output


pathmodelo = 'Introducir la ruta del modelo que se quiera probar'

env = gym.make("HumanoidStandup-v4", render_mode="rgb_array", camera_name="free")

env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True)

env = RecordEpisodeStatistics(env, deque_size=NUMEPISODIOS)

with open(pathmodelo, 'rb') as f:
    net = pickle.load(f)

for i in range(NUMEPISODIOS):
    obs = env.reset()
    totalreward = []
    for step in range(NUMSTEPS):
        if (step == 0):
            output = net.activate(obs[0])
        else:

            output = net.activate(new_obs)

        new_obs, reward, done, truncated, info = env.step(translate_output(output))
        totalreward.append(reward)

        time.sleep(0.001)

    print(np.sum(totalreward))
env.close()
