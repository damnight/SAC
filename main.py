import numpy as np
import matplotlib.pyplot as plt

def make_deterministic_average():
    #open files as np array
    det1 = np.loadtxt('Experiments/3 Variable Hyperparameters/Deterministic Avg/1 Hopper Deterministic/Hopper-v2-deter1.txt')
    det2 = np.loadtxt('Experiments/3 Variable Hyperparameters/Deterministic Avg/2 Hopper Deterministic/Hopper-v2-deter2.txt')
    det3 = np.loadtxt('Experiments/3 Variable Hyperparameters/Deterministic Avg/3 Hopper Deterministic/Hopper-v2-deter3.txt')

    #average arrays
    det_avg = np.mean( np.array([ det1, det2, det3 ]), axis=0 )
    a = open('Experiments/3 Variable Hyperparameters/Stochastic vs Deterministc/Hopper Average/Hopper-v2-det_avg.txt', 'w')
    np.savetxt(a, det_avg)
    a.close()

    return det_avg

def get_array(filepath):
    return np.loadtxt(filepath)


def display_graph_det_avg(dict):
    keys = list(dict.keys())
    print(type(keys))
    plt.plot(dict['deterministic average'], label=keys[0])
    plt.plot(dict['stochastic replay'], label=keys[1])
    plt.xlabel("steps")
    plt.ylabel("reward")
    plt.legend(loc='upper left')
    plt.title("Deterministic Average vs Stochastic Replay")
    plt.show()

if __name__ == "__main__":
    deter_avg = make_deterministic_average()
    stoch_replay = get_array('Experiments/3 Variable Hyperparameters/Stochastic vs Deterministc/Stochastic Replay/Hopper-v2-stoch_replay_deter.txt')
    a = { 'deterministic average' : np.array(deter_avg),
          'stochastic replay': np.array(stoch_replay)}
    display_graph_det_avg(a)