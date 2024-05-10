import pandas
import matplotlib.pyplot as plt
import io


def plotear(max):
    datagenomas = pandas.read_pickle("./historial-genomas.p", compression='infer')
    datageneraciones = pandas.read_pickle("./historial-generaciones.p", compression='infer')

    plt.switch_backend('agg')

    min_reward = datagenomas['Reward'].min()

    datagenomas.reset_index().plot(x='index', y='Reward', kind='line')
    plt.xlabel('Individuos')
    plt.ylabel('Reward')
    plt.title('Individuos vs. Reward')

    # Set the y-axis limit to start from 0 and end at 0
    plt.ylim(min_reward -1, max)

    # Save the plot as a PNG file
    plt.savefig('genomasplot.png', format='png')

    plt.close()

    min_reward = datageneraciones['Reward'].min()
    datageneraciones.reset_index().plot(x='index', y='Reward', kind='line')
    plt.xlabel('Generacion')
    plt.ylabel('Reward')
    plt.title('Generacion vs. Reward')

    # Set the y-axis limit to start from 0 and end at 0
    plt.ylim(min_reward -1 , max)

    # Save the plot as a PNG file
    plt.savefig('geneneracionesplot.png', format='png')

    plt.close()

    max_rewards_per_genoma = datagenomas.groupby('Genoma')['Reward'].max()

    # Plotting the highest reward per genoma without dots
    max_rewards_per_genoma.plot(kind='line', marker='', linestyle='-')
    plt.xlabel('Generation')
    plt.ylabel('Highest Reward')
    plt.title('Highest Reward per Generation')
    plt.grid(True)
    plt.xticks(rotation=45)

    # Save the plot as a PNG file
    plt.savefig('bestgenomas.png', format='png')

    # Display the plot
    plt.close()
