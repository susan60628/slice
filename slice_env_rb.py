import math
import sys
import numpy as np
import gc
import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt

class Slice:
    def __init__(self, iteration):
        self.iteration = iteration
        self.size_of_eMBB = [6400, 12800, 19200, 25600, 32000, 300000, 400000, 500000, 600000, 700000]
        self.column_of_rb = ["arrival", "datasize", "latency", "error", "snr"]
        self.action_space = self.define_action()
        self.n_actions = len(self.action_space)
        self.n_features = 3
        self.SE_hist = []
        self.latency_hist = []
        self.latency_uRLLC_hist = []
        self.latency_eMBB_hist = []
        self.latency_mMTC_hist = []
        np.savetxt("action_data.csv", self.action_space, fmt='%d,')

    def define_action(self):
        # define action for bandwidth allocation (10MHz for three services, the minimum unit is 1MHz)
        np_action_space_1 = np.zeros((36,3))
        i = 0
        for m in range(1, 9):
            for n in range(1, 10-m):
                np_action_space_1[i, :] = np.array([m, n, (10-m-n)])
                i += 1

        return np_action_space_1

    def round_robin(self, data, datarate): # data(byte), datarate(Mbps)
        # sort data by size (Z->A)
        data = -np.sort(-data, axis=0)

        # calculate the transmit datasize in 0.5ms by datarate
        datarate_ms = (datarate * 1000000) / (8 * 2000)

        # transmit data
        i = 0
        while i < 2000:
            data = data - (datarate_ms / data.shape[0])
            # skip the negitive and zero value
            data = data[data.min(axis=1)>0, :]
            i += 1
        
        return data

    def step(self, action, o, data_uRLLC, data_eMBB, data_mMTC, step):
        # get action space detail with chooses action
        np_action_detail = np.array(self.action_space[action:(action+1), :]).flatten()

        # calculate network capacity
        capacity_uRLLC = math.log2((pow(10,(20/10))+1)) * int(np_action_detail[0])
        capacity_eMBB = math.log2((pow(10,(12/10))+1)) * int(np_action_detail[1])
        capacity_mMTC = math.log2((pow(10,(-10/10))+1)) * int(np_action_detail[2])

        # calculate spectrum efficiency (bit/Hz)
        SE = (capacity_uRLLC + capacity_eMBB + capacity_mMTC) / 10
        # change SE dimension
        SE_point = SE / (6.65 - 0.14) * 2 - 1

        # calculate latency of spectrum: 8Mbps = 1 MB/s, 1000ms = 1s
        np_latency_uRLLC = (data_uRLLC[:, 1:2] * 1000 * 8) / (capacity_uRLLC * 1000000)
        np_latency_eMBB = (data_eMBB[:, 1:2] * 1000 * 8) / (capacity_eMBB * 1000000)
        np_latency_mMTC = (data_mMTC[:, 1:2] * 1000 * 8) / (capacity_mMTC * 1000000)
        np_latency_uRLLC_a = data_uRLLC[:, 2:3] - np_latency_uRLLC
        np_latency_eMBB_a = (data_eMBB[:, 2:3] - np_latency_eMBB) * 0.1
        np_latency_mMTC_a = (data_mMTC[:, 2:3] - np_latency_mMTC) * 0.001
        # change latency dimension  
        latency_point = (np.average(np_latency_uRLLC_a) + np.average(np_latency_eMBB_a) + np.average(np_latency_mMTC_a)) / 3

        # calculate error rate
        error_uRLLC = 0 if np.sum(data_uRLLC[:, 1:2]) <= (capacity_uRLLC * 1000000 / 8) else (self.round_robin(data_uRLLC[:, 1:2], capacity_uRLLC).shape[0])/(data_uRLLC[:, 1:2].shape[0])
        error_eMBB = 0 if np.sum(data_eMBB[:, 1:2]) <= (capacity_eMBB * 1000000 / 8) else (self.round_robin(data_eMBB[:, 1:2], capacity_eMBB).shape[0])/(data_eMBB[:, 1:2].shape[0])
        error_mMTC = 0 if np.sum(data_mMTC[:, 1:2]) <= (capacity_mMTC * 1000000 / 8) else (self.round_robin(data_mMTC[:, 1:2], capacity_mMTC).shape[0])/(data_mMTC[:, 1:2].shape[0])
        error_uRLLC_a = (0.00001 - error_uRLLC) * 10000
        error_eMBB_a = (0.01 - error_eMBB) * 10
        error_mMTC_a = 0.1 - error_mMTC
        # change error rate dimension
        error_point = (error_uRLLC_a + error_eMBB_a + error_mMTC_a) / 3

        # append list_hist if RL learned
        if step > self.iteration and (step % 5 == 0):           
            self.SE_hist.append(SE)
            self.latency_hist.append(np.average(np_latency_uRLLC) + np.average(np_latency_eMBB) + np.average(np_latency_mMTC))
            self.latency_uRLLC_hist.append(np.average(np_latency_uRLLC))
            self.latency_eMBB_hist.append(np.average(np_latency_eMBB))
            self.latency_mMTC_hist.append(np.average(np_latency_mMTC))

        reward = SE_point + latency_point + error_point
        return reward

    def reset(self):
        # for state t+1
        i_temp = self.iteration + 1
        
        # generate random packet for iteration
        self.np_data_uRLLC = self.generate_data("uRLLC",8,0,int((i_temp*1000)/8),1,0.00001)
        self.np_data_eMBB = self.generate_data("eMBB",25,0,int((i_temp*1000)/25),10,0.001)
        self.np_data_mMTC = self.generate_data("mMTC",25,0,int((i_temp*1000)/25),1000,0.01)

        # cut data to iteration
        self.np_data_uRLLC_a = self.cut_data(self.np_data_uRLLC[:, :1], i_temp)
        self.np_data_eMBB_a = self.cut_data(self.np_data_eMBB[:, :1], i_temp)
        self.np_data_mMTC_a = self.cut_data(self.np_data_mMTC[:, :1], i_temp)

        return self.np_data_uRLLC_a, self.np_data_eMBB_a, self.np_data_mMTC_a, self.np_data_uRLLC, self.np_data_eMBB, self.np_data_mMTC

    def result_to_dataFrame(self, result, cols):
        self.data = DataFrame(result, columns=cols)
        #print(self.data)
        return self.data

    def generate_data(self, type, min_a, max_a, s, latency, error):
        # defined arrival time (ms), max_a=0 -> fix arrival time
        if max_a == 0:
            self.np_arrival = np.full(s, min_a)
        else:
            self.np_arrival = np.random.randint(min_a, max_a, [s])

        # defined datasize (byte)
        if type == "mMTC":
            # self.np_datasize = np.full(s, 40)
            self.np_datasize = np.random.randint(100,250,[s])
        elif type == "eMBB" :
            # self.np_datasize = np.random.randint(100,250,[s])
            self.np_datasize = np.random.choice(self.size_of_eMBB,s)
        else: # uRLLC
            # self.np_datasize = np.random.choice(self.size_of_eMBB,s)
            self.np_datasize = np.full(s, 40)
        
        # defined SLA latency (ms)
        self.np_latency = np.full(s, latency)

        # defined SLA error rate (error packet num / total packet num)
        self.np_error = np.full(s, error)

        # combined data
        self.np_data = np.hstack((self.np_arrival[:,np.newaxis], self.np_datasize[:,np.newaxis], self.np_latency[:,np.newaxis], self.np_error[:,np.newaxis]))

        return self.np_data

    def cut_data(self, np_data, iteration):
        self.np_flatten = np.array(np_data).flatten()
        self.count_x = 0
        self.count_y = 0
        self.j = 0
        self.second = 1000
        self.np_arr = np.zeros((iteration,2))
        for i in range(self.np_flatten.size):
            self.second = self.second - self.np_flatten[i]
            if self.second > 0:
                self.count_y += 1
            elif self.second == 0:
                self.np_arr[self.j][0]= self.count_x
                self.np_arr[self.j][1]= self.count_y + 1
                self.j += 1
                self.second += 1000
                self.count_x = i + 1
                self.count_y = 0
            else:
                self.np_arr[self.j][0]= self.count_x
                self.np_arr[self.j][1]= self.count_y
                self.j += 1
                self.second += 1000
                self.count_x = i
                self.count_y = 1
            if self.j == iteration:
                break
        return self.np_arr

    def plot_se(self):
        plt.plot(np.arange(len(self.SE_hist)), self.SE_hist)
        plt.ylabel('SE')
        plt.xlabel('training steps')
        plt.show()

    def plot_latency(self):
        plt.plot(np.arange(len(self.latency_hist)), self.latency_hist, label="Total")
        plt.plot(np.arange(len(self.latency_uRLLC_hist)), self.latency_uRLLC_hist, label="uRLLC")
        plt.plot(np.arange(len(self.latency_eMBB_hist)), self.latency_eMBB_hist, label= "eMBB")
        plt.plot(np.arange(len(self.latency_mMTC_hist)), self.latency_mMTC_hist, label="mMTC")
        plt.ylabel('Latency')
        plt.xlabel('training steps')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    slice = Slice(1000)