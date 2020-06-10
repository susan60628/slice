import math
import sys
import numpy as np
import gc
import pandas
from pandas import DataFrame

size_of_uRLLC = [6400, 12800, 19200, 25600, 32000]
column_of_data = ["arrival", "datasize", "latency", "error", "snr"]

def trans_SNR(SNR_b):
    SNR_a = pow(10,(SNR_b/10))
    return SNR_a

def result_to_dataFrame(result, cols):
    data = DataFrame(result, columns=cols)
    print(data)
    return data

def generate_data(type, min_a, max_a, s, latency, error):
    np_arrival = np.random.randint(min_a, max_a, [s])
    if type == "mMTC":
        np_datasize = np.full(s, 40)
        np_snr = np.full(s, 0.194)
    elif type == "eMBB" :
        np_datasize = np.random.randint(100,250,[s])
        np_snr = np.full(s, 3162)
    else:
        np_datasize = np.random.choice(size_of_uRLLC,s)
        np_snr = np.full(s, 1000)
    np_latency = np.full(s, latency)
    np_error = np.full(s, error)
    np_data = np.hstack((np_arrival[:,np.newaxis], np_datasize[:,np.newaxis], np_latency[:,np.newaxis], np_error[:,np.newaxis], np_snr[:,np.newaxis]))
    del np_arrival, np_datasize, np_latency, np_error
    gc.collect()
    return np_data

def cut_data(np_data):
    np_flatten = np.array(np_data).flatten()
    print(np_flatten)
    count_x = 0
    count_y = 0
    j = 0
    second = 100
    arr = [[0]*2 for i in range(8000)]
    for i in range(np_flatten.size):
        second = second - np_flatten[i]
        if second > 0:
            count_y += 1
        elif second == 0:
            arr[j][0]= count_x
            arr[j][1]= count_y + 1
            j += 1
            second += 100
            count_x = i + 1
            count_y = 0
        else:
            arr[j][0]= count_x
            arr[j][1]= count_y
            j += 1
            second += 100
            count_x = i
            count_y = 1
        if j == 8000:
            break
    return arr         

if __name__ == "__main__":
    #np_data_uRLLC = generate_data("uRLLC",50,99,16000,1,0.00001)
    #np_data_eMBB = generate_data("eMBB",5,12,160000,10,0.001)
    np_data_mMTC = generate_data("mMTC",1,99,800000,500,0.01)
    #df_data_uRLLC = result_to_dataFrame(np_data_uRLLC, column_of_data)
    #df_data_eMBB = result_to_dataFrame(np_data_eMBB, column_of_data)
    df_data_mMTC = result_to_dataFrame(np_data_mMTC, column_of_data)
    #arr_data_uRLLC = cut_data(np_data_uRLLC[:, :1])
    #arr_data_eMBB = cut_data(np_data_eMBB[:, :1])
    arr_data_mMTC = cut_data(np_data_mMTC[:, :1])