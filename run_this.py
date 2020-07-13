from slice_env_rb import Slice
from DQN_modified import DeepQNetwork
import numpy as np
import matplotlib.pyplot as plt

iteration = 100
list_action = []
b_draw_plt = True

def plot_action():
    plt.plot(np.arange(len(list_action)), list_action)
    plt.ylabel('Action')
    plt.xlabel('training steps')
    plt.show()

def run_Slice():
    step = 0
    for episode in range(80):
        # initial observation
        np_uRLLC, np_eMBB, np_mMTC, np_uRLLC_data, np_eMBB_data, np_mMTC_data = env.reset()

        for itera_ in range(iteration):

            # defined observation
            observation = np.zeros((3,))
            observation[0] = np_uRLLC[itera_][1]
            observation[1] = np_eMBB[itera_][1]
            observation[2] = np_mMTC[itera_][1]

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get reward
            reward = env.step(action,
                              observation,
                              np_uRLLC_data[int(np_uRLLC[itera_][0]):int(np_uRLLC[itera_][0] + np_uRLLC[itera_][1]), :],
                              np_eMBB_data[int(np_eMBB[itera_][0]):int(np_eMBB[itera_][0] + np_eMBB[itera_][1]), :],
                              np_mMTC_data[int(np_mMTC[itera_][0]):int(np_mMTC[itera_][0] + np_mMTC[itera_][1]), :],
                              step)

            # defined next observation
            observation_ = np.zeros((3,))
            observation_[0] = np_uRLLC[itera_+1][1]
            observation_[1] = np_eMBB[itera_+1][1]
            observation_[2] = np_mMTC[itera_+1][1]

            RL.store_transition(observation, action, reward, observation_)

            if (step > iteration) and (step % 5 == 0):
                cost = RL.learn()
                print("episode:", episode, "iteration:", step,"o:", observation, "a:", action, "r:", reward, "o_:", observation_, "cost:", cost)
                list_action.append(action)

            step += 1

    # end of game
    print('game over')

if __name__ == "__main__":
    # Slice game
    env = Slice(iteration)
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.001,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=1000,
                      memory_size=10000,
                      #output_graph=True
                      )
    run_Slice()
    if b_draw_plt:
        RL.plot_cost()
        env.plot_latency()
        env.plot_se()
        # env.plot_cr()
        plot_action()