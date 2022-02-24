import numpy as np
import matplotlib.pyplot as plt

def make_deterministic_average():
    #open files as np array
    det1 = np.loadtxt('Experiments/3 Variable Hyperparameters/Deterministic Avg/1 Hopper Deterministic/Hopper-v2-deter1.txt')
    det2 = np.loadtxt('Experiments/3 Variable Hyperparameters/Deterministic Avg/2 Hopper Deterministic/Hopper-v2-deter2.txt')
    det3 = np.loadtxt('Experiments/3 Variable Hyperparameters/Deterministic Avg/3 Hopper Deterministic/Hopper-v2-deter3.txt')

    #average arrays
    det_avg = np.mean( np.array([ det1, det2, det3 ]), axis=0 )
    a = open('Experiments/3 Variable Hyperparameters/Stochastic vs Deterministc/Deterministic Hopper Baseline/Hopper-v2-det_avg.txt', 'w')
    np.savetxt(a, det_avg)
    a.close()

    return det_avg

def get_array(filepath):
    return np.loadtxt(filepath)

def make_det_avg():
    deter_avg = make_deterministic_average()
    stoch_replay = get_array('Experiments/3 Variable Hyperparameters/Stochastic vs Deterministc/Stochastic Hopper with Deterministic Replay/Hopper-v2-stoch_replay_deter.txt')
    a = { 'deterministic average' : np.array(deter_avg),
          'stochastic replay': np.array(stoch_replay)}
    display_graph_det_avg(a)

def display_graph_det_avg(dict):
    keys = list(dict.keys())
    print(type(keys))
    plt.plot(dict['deterministic average'], label=keys[0])
    plt.plot(dict['stochastic replay'], label=keys[1])
    plt.xlabel("steps")
    plt.ylabel("reward")
    plt.legend(loc='upper left')
    #plt.title("Deterministic Average vs Stochastic Replay")
    plt.savefig('Experiments/Figures/det_avg_stoch_replay.png', bbox_inches='tight')
    #plt.show()


def make_default():
    ant_def = get_array('Experiments/1 Default Hyperparameters/Ant Default/Ant_Default.txt')
    cheetah_def = get_array('Experiments/1 Default Hyperparameters/HalfCheetah Default/HalfCheetah Default.txt')
    hopper_def  = get_array('Experiments/1 Default Hyperparameters/Hopper Default/Hopper Default.txt')
    humanoid_def = get_array('Experiments/1 Default Hyperparameters/Humanoid Default/Humanoid-v2 Default.txt')
    walker2d_def = get_array('Experiments/1 Default Hyperparameters/Walker2d Default/Walker2d Default.txt')

    plt.plot(ant_def, label='Ant', color='y')
    plt.plot(cheetah_def, label='Half Cheetah', color='m')
    plt.plot(hopper_def, label='Hopper', color='r')
    plt.plot(humanoid_def, label='Humanoid', color='g')
    plt.plot(walker2d_def, label='Walker 2D', color='b')

    plt.xlabel("steps")
    plt.ylabel("reward")
    plt.legend(loc='upper left')
    #plt.title("SAC Default Hyperparameters")
    plt.savefig('Experiments/Figures/sac_default_params.png', bbox_inches='tight')
    #plt.show()

def make_stoch_vs_det():
    b1 = get_array('Experiments/3 Variable Hyperparameters/HopperBaselines/1/Hopper-v2-baseline1.txt')
    b2 = get_array('Experiments/3 Variable Hyperparameters/HopperBaselines/2/Hopper-v2-baseline2.txt')
    b3 = get_array('Experiments/3 Variable Hyperparameters/HopperBaselines/3/Hopper-v2-baseline3.txt')
    b4 = get_array('Experiments/3 Variable Hyperparameters/HopperBaselines/4/Hopper-v2-baseline4.txt')
    b5 = get_array('Experiments/3 Variable Hyperparameters/HopperBaselines/5/Hopper-v2-baseline5.txt')

    #average arrays
    stoch_avg = np.mean( np.array([ b1, b2, b3, b4, b5]), axis=0 )
    a = open('Experiments/3 Variable Hyperparameters/Stochastic vs Deterministc/Stochastic Hopper Baseline/Hopper-v2-stoch_avg.txt', 'w')
    np.savetxt(a, stoch_avg)
    a.close()

    det_avg = get_array('Experiments/3 Variable Hyperparameters/Stochastic vs Deterministc/Deterministic Hopper Baseline/Hopper-v2-det_avg.txt')

    plt.plot(stoch_avg, label='stochastic average', color='r')
    plt.plot(det_avg, label='deterministic average', color='c')
    plt.xlabel("steps")
    plt.ylabel("reward")
    plt.legend(loc='upper left')
    #plt.title("stochastic vs deterministic training")
    plt.savefig('Experiments/Figures/stoch_vs_det_train.png', bbox_inches='tight')
    #plt.show()

def plot_array():
    a = get_array('Experiments/3 Variable Hyperparameters/HopperBaselines/1/Hopper-v2-baseline1.txt')
    plt.plot(a, color='r')
    plt.show()

def make_reward():
    r1 = get_array('Experiments/3 Variable Hyperparameters/Reward/Reward 1/Hopper-v2-r1.txt')
    r3 = get_array('Experiments/3 Variable Hyperparameters/Reward/Reward 3/Hopper-v2-r3.txt')
    r10 = get_array('Experiments/3 Variable Hyperparameters/Reward/Reward 10/Hopper-v2-r10.txt')
    r30 = get_array('Experiments/3 Variable Hyperparameters/Reward/Reward 30/Hopper-v2-r30.txt')
    r100 = get_array('Experiments/3 Variable Hyperparameters/Reward/Reward 100/Hopper-v2-r100.txt')

    plt.plot(r1, label='Reward 1', color='y')
    plt.plot(r3, label='Reward 3', color='m')
    plt.plot(r10, label='Reward 10', color='r')
    plt.plot(r30, label='Reward 30', color='g')
    plt.plot(r100, label='Reward 100', color='b')

    plt.xlabel("steps")
    plt.ylabel("reward")
    plt.legend(loc='upper left')
    #plt.title("Reward Scale")
    plt.savefig('Experiments/Figures/reward_scale.png', bbox_inches='tight')
    #plt.show()

def make_tau():
    t1 = get_array('Experiments/3 Variable Hyperparameters/Tau/Tau 1/Hopper-v2-t1.txt')
    t2 = get_array('Experiments/3 Variable Hyperparameters/Tau/Tau 2Hopper-v2-t2.txt')
    t3 = get_array('Experiments/3 Variable Hyperparameters/Tau/Tau 3/Hopper-v2-t3.txt')
    t4 = get_array('Experiments/3 Variable Hyperparameters/Tau/Tau 4/Hopper-v2-t4.txt')

    plt.plot(t1, label='Reward 1', color='y')
    plt.plot(t2, label='Reward 3', color='m')
    plt.plot(t3, label='Reward 10', color='r')
    plt.plot(t4, label='Reward 30', color='g')


    plt.xlabel("steps")
    plt.ylabel("reward")
    plt.legend(loc='upper left')
    #plt.title("Tau Coefficient")
    plt.savefig('Experiments/Figures/tau_coefficient.png', bbox_inches='tight')
    #plt.show()

if __name__ == "__main__":
    #plot_array()
    #make_det_avg()
    #make_default()
    #make_stoch_vs_det()
    #make_reward()
    #make_tau()
