import math
import sys
import numpy as np
import gc
import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt

class Slice:
    def __init__(self):
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

    def define_action(self): # Action Shape: (26244, 27) => 9^3 * 36 = 26244
        # define action for bandwidth allocation (10MHz for three services, the minimum unit is 1MHz)
        np_action_space_1 = np.zeros((36,3))
        i = 0
        for m in range(1, 9):
            for n in range(1, 10-m):
                np_action_space_1[i, :] = np.array([m, n, (10-m-n)])
                i += 1
        # copy 729 times to fit action 2
        np_action_space_1_c = np.copy(np_action_space_1)
        for m in range(728):
            np_action_space_1_c = np.vstack((np_action_space_1_c, np_action_space_1))
        np_action_space_1_s = np_action_space_1_c[np.lexsort(np.fliplr(np_action_space_1_c).T)]
        
        # define action for VNF deployment (1: central cloud, 0: edge cloud)
        np_action_space_2 = np.ones((9,8))
        for m in range(9):
            np_action_space_2[m, :m] = 0
        # copy once 9^2
        np_action_space_2_c = np.copy(np_action_space_2)
        for m in range(8):
            np_action_space_2_c = np.vstack((np_action_space_2_c, np_action_space_2))
        np_action_space_2_s = np_action_space_2_c[np.lexsort(np.fliplr(np_action_space_2_c).T)]
        np_action_space_2_n = np.hstack((np_action_space_2_s, np_action_space_2_c))
        # copy twice 9^3
        np_action_space_2_n_c = np.copy(np_action_space_2_n)
        np_action_space_2_c_c = np.copy(np_action_space_2_c)
        for m in range(8):
            np_action_space_2_n_c = np.vstack((np_action_space_2_n_c, np_action_space_2_n))
            np_action_space_2_c_c = np.vstack((np_action_space_2_c_c, np_action_space_2_c))
        np_action_space_2_n_s = np_action_space_2_n_c[np.lexsort(np.fliplr(np_action_space_2_n_c).T)]
        np_action_space_2_n_n = np.hstack((np_action_space_2_n_s, np_action_space_2_c_c))
        # copy 36 times to fit action 1
        np_action_space_2_n_n_c = np.copy(np_action_space_2_n_n)
        for m in range(35):
            np_action_space_2_n_n_c = np.vstack((np_action_space_2_n_n_c, np_action_space_2_n_n))
       
        # join two action
        np_action_space = np.concatenate((np_action_space_1_s, np_action_space_2_n_n_c), axis=1)

        # del temp
        del np_action_space_1, np_action_space_1_c, np_action_space_1_s, np_action_space_2, np_action_space_2_c, np_action_space_2_s, np_action_space_2_n, np_action_space_2_n_c, np_action_space_2_c_c, np_action_space_2_n_s, np_action_space_2_n_n, np_action_space_2_n_n_c
        gc.collect()

        return np_action_space

    def round_robin(self, data, datarate): # data(byte), datarate(Mbps)
        # sort data by size (Z->A)
        data = -np.sort(-data, axis=0)

        # calculate the transmit datasize in 0.5ms by datarate
        datarate_ms = (datarate * 1000000) / (8 * 200)

        # transmit data
        i = 0
        while i < 200:
            data = data - (datarate_ms / data.shape[0])
            # skip the negitive and zero value
            data = data[data.min(axis=1)>0, :]
            i += 1
        
        return data

    def computational_rate(self, list_latency, np_action, rb, mcs):
        # computational requirement with Cexp = 550 GFLOPS, polynomial coefficients = (32.583, 1.072, 0.03), CPU = 2GHz, d_e = 30m, d_ec = 60m, d_c = 90m
        c = (550 * rb * 33.685 * mcs * 0.00001) / 4
        c_e = 0
        c_c = 0
        for i in range(8):
            c_temp1 = 0
            c_temp2 = 0
            if np_action[i] == 0: # in edge cloud
                if i == 0: # first VNF is in edge, d_e = 30m
                    c_temp1 = c / (list_latency[i] - 0.0015)
                elif i < 7 and np_action[i+1] != np_action[i]: # split exist, d_ec = 60m
                    c_temp1 = c / list_latency[i]
                    c_temp2 = c / (list_latency[i+1] - 0.003)
                    c_temp1 = max(c_temp1, c_temp2)
                else:
                    c_temp1 = c / list_latency[i]
                c_e += c_temp1
            else: # in central cloud
                if i == 1: # first VNF is in central, d_c = 90m
                    c_temp1 = c / (list_latency[i] - 0.0045)
                elif np_action[i-1] != np_action[i]: # split exist, d_ec = 60m
                    c_temp1 = c / list_latency[i]
                    c_temp2 = c / (list_latency[i-1] - 0.003)
                    c_temp1 = max(c_temp1, c_temp2)
                else:
                    c_temp1 = c / list_latency[i]
                c_c += c_temp1
        
        return c_e, c_c

    def step(self, action, o, data_uRLLC, data_eMBB, data_mMTC):
        # get action space detail with chooses action
        np_action_detail = np.array(self.action_space[action:(action+1), :]).flatten()

        # calculate network capacity(float)
        capacity_uRLLC = math.log2((pow(10,(20/10))+1)) * int(np_action_detail[0])
        capacity_eMBB = math.log2((pow(10,(12/10))+1)) * int(np_action_detail[1])
        capacity_mMTC = math.log2((pow(10,(-10/10))+1)) * int(np_action_detail[2])

        # calculate spectrum efficiency (bit/Hz)
        SE = (capacity_uRLLC + capacity_eMBB + capacity_mMTC) / 10
        # change SE dimension
        SE_point = SE / (6.65 - 0.14)
        self.SE_hist.append(SE)

        # calculate latency of spectrum: 8Mbps = 1 MB/s, 100ms = 1s
        np_latency_uRLLC = (data_uRLLC[:, 1:2] * 100 * 8) / (capacity_uRLLC * 1000000)
        np_latency_eMBB = (data_eMBB[:, 1:2] * 100 * 8) / (capacity_eMBB * 1000000)
        np_latency_mMTC = (data_mMTC[:, 1:2] * 100 * 8) / (capacity_mMTC * 1000000)
        np_latency_uRLLC_a = data_uRLLC[:, 2:3] - np_latency_uRLLC
        np_latency_eMBB_a = (data_eMBB[:, 2:3] - np_latency_eMBB) * 0.1
        np_latency_mMTC_a = (data_mMTC[:, 2:3] - np_latency_mMTC) * 0.01
        # change latency dimension
        latency_point = (np.sum(np_latency_uRLLC_a) + np.sum(np_latency_eMBB_a) + np.sum(np_latency_mMTC_a)) / (np_latency_uRLLC_a.shape[0] + np_latency_eMBB_a.shape[0] + np_latency_mMTC_a.shape[0])

        # calculate error rate
        error_uRLLC = 0 if np.sum(data_uRLLC[:, 1:2]) <= (capacity_uRLLC * 1000000 / 8) else (self.round_robin(data_uRLLC[:, 1:2], capacity_uRLLC).shape[0])/(data_uRLLC[:, 1:2].shape[0])
        error_eMBB = 0 if np.sum(data_eMBB[:, 1:2]) <= (capacity_eMBB * 1000000 / 8) else (self.round_robin(data_eMBB[:, 1:2], capacity_eMBB).shape[0])/(data_eMBB[:, 1:2].shape[0])
        error_mMTC = 0 if np.sum(data_mMTC[:, 1:2]) <= (capacity_mMTC * 1000000 / 8) else (self.round_robin(data_mMTC[:, 1:2], capacity_mMTC).shape[0])/(data_mMTC[:, 1:2].shape[0])
        error_uRLLC_a = (0.00001 - error_uRLLC) * 10000
        error_eMBB_a = (0.01 - error_eMBB) * 10
        error_mMTC_a = 0.1 - error_mMTC
        # change error rate dimension
        error_point = (error_uRLLC_a + error_eMBB_a + error_mMTC_a) / 3

        # get VNF deployment action
        np_action_uRLLC = np_action_detail[0:8]
        np_action_eMBB = np_action_detail[8:16]
        np_action_mMTC = np_action_detail[16:24]
        # set latency constraints
        list_latency_uRLLC = [0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        list_latency_eMBB = [1, 3, 3, 3, 10, 10, 10, 10]
        list_latency_mMTC = [10, 10, 10, 10, 200, 500, 1000, 2000]
        # calculate resource blocks(? MHz * 1000 / 180 kHz)
        rb_uRLLC = int(np_action_detail[0] * 1000 / 180)
        rb_eMBB = int(np_action_detail[1] * 1000 / 180)
        rb_mMTC = int(np_action_detail[2] * 1000 / 180)       
        # calculate computational rate
        c_c_uRLLC, c_e_uRLLC = self.computational_rate(list_latency_uRLLC, np_action_uRLLC, rb_uRLLC, 27)
        c_c_eMBB, c_e_eMBB = self.computational_rate(list_latency_eMBB, np_action_eMBB, rb_eMBB, 27)
        c_c_mMTC, c_e_mMTC = self.computational_rate(list_latency_mMTC, np_action_mMTC, rb_mMTC, 13)
        # change computational rate dimension
        c_c_total = c_c_uRLLC + c_c_eMBB + c_c_mMTC
        c_e_total = c_e_uRLLC + c_e_eMBB + c_e_mMTC
        c_c_point = (c_c_total / 8960) * -1
        c_e_point = (c_e_total / 4480) * -1
        
        # hist of latency
        self.latency_hist.append(np.average(np_latency_uRLLC) + np.average(np_latency_eMBB) + np.average(np_latency_mMTC) + (0.75 / np_latency_uRLLC_a.shape[0]) + (50 / np_latency_eMBB_a.shape[0]) + (3740 / np_latency_mMTC_a.shape[0]))
        self.latency_uRLLC_hist.append(np.average(np_latency_uRLLC) + (0.75 / np_latency_uRLLC_a.shape[0]))
        self.latency_eMBB_hist.append(np.average(np_latency_eMBB) + (50 / np_latency_eMBB_a.shape[0]))
        self.latency_mMTC_hist.append(np.average(np_latency_mMTC) + (3740 / np_latency_mMTC_a.shape[0]))

        reward = SE_point + latency_point + error_point + c_c_point + c_e_point
        return reward

    def reset(self, iteration):
        # for state t+1
        i_temp = iteration + 1
        
        # generate random packet for iteration
        self.np_data_uRLLC = self.generate_data("uRLLC",20,80,int((i_temp*100)/20),1,0.00001)
        self.np_data_eMBB = self.generate_data("eMBB",2,10,int((i_temp*100)/2),10,0.001)
        self.np_data_mMTC = self.generate_data("mMTC",10,20,int((i_temp*100)/10),100,0.01)

        # arrange numpy data to dataframe
        # self.df_data_uRLLC = self.result_to_dataFrame(self.np_data_uRLLC, self.column_of_rb)
        # self.df_data_eMBB = self.result_to_dataFrame(self.np_data_eMBB, self.column_of_rb)
        # self.df_data_mMTC = self.result_to_dataFrame(self.np_data_mMTC, self.column_of_rb)

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
        # defined arrival time (ms)
        self.np_arrival = np.random.randint(min_a, max_a, [s])

        # defined datasize (byte)
        if type == "mMTC":
            self.np_datasize = np.random.randint(100,250,[s])
        elif type == "eMBB" :
            self.np_datasize = np.random.choice(self.size_of_eMBB,s)
        else: # uRLLC
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
        self.second = 100
        self.np_arr = np.zeros((iteration,2))
        for i in range(self.np_flatten.size):
            self.second = self.second - self.np_flatten[i]
            if self.second > 0:
                self.count_y += 1
            elif self.second == 0:
                self.np_arr[self.j][0]= self.count_x
                self.np_arr[self.j][1]= self.count_y + 1
                self.j += 1
                self.second += 100
                self.count_x = i + 1
                self.count_y = 0
            else:
                self.np_arr[self.j][0]= self.count_x
                self.np_arr[self.j][1]= self.count_y
                self.j += 1
                self.second += 100
                self.count_x = i
                self.count_y = 1
            if self.j == iteration:
                break
        return self.np_arr
    
    def plot_latency(self):
        plt.plot(np.arange(len(self.latency_hist)), self.latency_hist, label="Total")
        plt.plot(np.arange(len(self.latency_uRLLC_hist)), self.latency_uRLLC_hist, label="uRLLC")
        plt.plot(np.arange(len(self.latency_eMBB_hist)), self.latency_eMBB_hist, label= "eMBB")
        plt.plot(np.arange(len(self.latency_mMTC_hist)), self.latency_mMTC_hist, label="mMTC")
        plt.ylabel('Latency')
        plt.xlabel('training steps')
        plt.show()

    def plot_se(self):
        plt.plot(np.arange(len(self.SE_hist)), self.SE_hist)
        plt.ylabel('SE')
        plt.xlabel('training steps')
        plt.show()

if __name__ == "__main__":
    slice = Slice()