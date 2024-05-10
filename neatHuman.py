import time
import gym
import os
import neat
import pandas
import plots
import numpy as np
import pickle

GENERACIONES = 300  # Nº de generaciones
CHECKPOINTS = 1  # Cada cuanto guardar un checkpoint de la generación actual
STEPS = 300  # Nº de acciones por genoma

env = gym.make("HumanoidStandup-v4", render_mode="human")

try:
    datagenomas = pandas.read_pickle("./historial-genomas.p", compression='infer')
    datageneraciones = pandas.read_pickle("./historial-generaciones.p", compression='infer')
except:
    data = {'Generacion': [], 'Reward': []}
    datageneraciones = pandas.DataFrame(data)
    data = {'Genoma': [], 'Reward': []}
    datagenomas = pandas.DataFrame(data)


def saveifhighest(net, reward):
    maxglobal = datagenomas.to_numpy().max()
    if (reward > maxglobal):
        with open("models/reward-" + str(int(reward)) + ".pkl", "wb") as f:
            pickle.dump(net, f)
            f.close()


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


#función para agregar todos los rewards obtenidos de las acciones del episodio
def rewardgenoma(rewards):
    mediareward = np.mean(rewards)
    maxreward = max(rewards)
    rewardfinal = (mediareward + maxreward) / 2

    return rewardfinal


def eval_genomes(genomes, config):
    rewardgeneration = 0
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.feed_forward.FeedForwardNetwork.create(genome, config)
        obs = env.reset()
        totalreward = []
        for step in range(STEPS):
            if (step == 0):
                output = net.activate(obs[0])
            else:

                output = net.activate(new_obs)

            new_obs, reward, done, truncated, info = env.step(translate_output(output))

            totalreward.append(reward)

            time.sleep(0.001)

        rewardfinal = rewardgenoma(totalreward)
        print("REWARD TOTAL: ", rewardfinal)
        if len(datagenomas) > 0:
            saveifhighest(net, rewardfinal)
        genome.fitness = rewardfinal
        rewardgeneration += rewardfinal
        datagenomas.loc[len(datagenomas)] = [len(datageneraciones), rewardfinal]
    datageneraciones.loc[len(datageneraciones)] = [len(datageneraciones), rewardgeneration / len(genomes)]
    datageneraciones.to_pickle('./historial-generaciones.p')
    datagenomas.to_pickle('./historial-genomas.p')
    plots.plotear(datagenomas.to_numpy().max())


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Population(config)
    # p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-299")

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(CHECKPOINTS))

    p.run(eval_genomes, GENERACIONES)


directorio = os.path.dirname(__file__)
config_path = os.path.join(directorio, 'config-feedforward')
run(config_path)
