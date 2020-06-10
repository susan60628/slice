from slice_env import Slice
from DQN_modified import DeepQNetwork
import numpy as np

iteration = 100

def run_Slice():
    step = 0
    for episode in range(500):
        # initial observation
        np_uRLLC, np_eMBB, np_mMTC, np_uRLLC_data, np_eMBB_data, np_mMTC_data = env.reset(iteration)

        for itera_ in range(iteration):

            # defined observation
            observation = np.zeros((3,))
            observation[0] = np_uRLLC[itera_][1] #test
            observation[1] = np_eMBB[itera_][1] #test
            observation[2] = np_mMTC[itera_][1]

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get reward
            reward = env.step(action,
                              observation,
                              np_uRLLC_data[int(np_uRLLC[itera_][0]):int(np_uRLLC[itera_][0] + np_uRLLC[itera_][1]), :],
                              np_eMBB_data[int(np_eMBB[itera_][0]):int(np_eMBB[itera_][0] + np_eMBB[itera_][1]), :],
                              np_mMTC_data[int(np_mMTC[itera_][0]):int(np_mMTC[itera_][0] + np_mMTC[itera_][1]), :],)

            # defined next observation
            observation_ = np.zeros((3,))
            observation_[0] = np_uRLLC[itera_+1][1]
            observation_[1] = np_eMBB[itera_+1][1]
            observation_[2] = np_mMTC[itera_+1][1]

            print("o:", observation, "a:", action, "r:", reward, "o_:", observation_)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            step += 1

    # end of game
    print('game over')

if __name__ == "__main__":
    # Slice game
    env = Slice()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      #output_graph=True
                      )
    run_Slice()
    RL.plot_cost()
    env.plot_latency()
    env.plot_se()