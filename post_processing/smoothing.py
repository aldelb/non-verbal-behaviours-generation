import pandas as pd
import os
from os.path import isfile
import numpy as np
import matplotlib.pyplot as plt
import sys

dirs = ["generation/data/output/27-07-2022_mosi_2/epoch_9/"]
figures = sys.argv[1] # "true" if you want to visualize the effect of smoothing in graph, "false" otherwise

def smooth(data, kernel):
    # lissage de la courbe par convolution (ici fenetre de taille 15)
    kernel_size = kernel# impair
    smoothed = np.convolve(data, np.ones(kernel_size)/kernel_size, 'valid')
    # complete les bords qui manquent avec les premieres/dernieres valeurs
    smoothed2 = [smoothed[0]]*int(kernel_size/2) + list(smoothed) + [smoothed[-1]]*int(kernel_size/2)
    smoothed = np.array(smoothed2)
    return smoothed


def plot_figure(x, y, y_smoothed, column, dir_figures, index):
    if(figures == "true"):
        fig = plt.figure(dpi=100)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(x, y, label="initial")
        ax1.plot(x, y_smoothed, label="smoothed")
        ax1.legend()
        plt.savefig(dir_figures + column + "_" + index + '_smoothed.png')
        plt.close()


#regul one file only
from scipy.signal import savgol_filter

sourcil = ["AU01_r", "AU02_r", "AU04_r"]
visage = ["AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r"]
bouche = ["AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r"]
clignement = ["AU45_r"]
rotation = ['pose_Rx', 'pose_Ry', 'pose_Rz']
gaze = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z', 'gaze_angle_x', 'gaze_angle_y']


columns_1 = gaze
columns_2 = rotation
columns_3 =  bouche
columns_4 = clignement
columns_5 =  sourcil

# if CGAN model
gaze_value = 2

# if AED model
# gaze_value = 1.2

for dir in dirs:
    for file in os.listdir(dir) :
        if(isfile(dir + file) and  ".csv" in file):
            current_file = dir + file
            dir_smoothed = dir +  "smoothed/"
            dir_figures = dir_smoothed+"figures/"
            print(current_file)
            if(not os.path.isdir(dir_smoothed)):
                os.mkdir(dir_smoothed)
            if(not os.path.isdir(dir_figures)):
                os.mkdir(dir_figures)
            df = pd.read_csv(current_file)
            for column in columns_1:
                initial_data = df[column].copy()
                df[column] = savgol_filter(df[column], 31, 7)
                plot_figure(range(len(df['timestamp'])), initial_data, df[column], column, dir_figures, file[0:-4])
            for column in columns_2:
                initial_data = df[column].copy()
                df[column] = savgol_filter(df[column], 71, 7)
                plot_figure(range(len(df['timestamp'])), initial_data, df[column], column, dir_figures, file[0:-4])
            for column in columns_3:
                initial_data = df[column].copy()
                df[column] = savgol_filter(df[column], 11, 7)
                plot_figure(range(len(df['timestamp'])), initial_data, df[column], column, dir_figures, file[0:-4])
            for column in columns_4:
                initial_data = df[column].copy()
                df[column] = df[column].apply(lambda x: 0 if x < gaze_value else 4)
                for i in range(len(df[column])):
                    if(df[column][i] == 4):
                        if(i!=0):
                            df[column][i-1] = 2.5
                        if(i<len(df[column])-1):
                            df[column][i+1] = 2.5
                        df[column][i+2:i+100] = 0                  
                plot_figure(range(len(df['timestamp'])), initial_data, df[column], column, dir_figures, file[0:-4])
            for column in columns_5:
                initial_data = df[column].copy()
                df[column] = savgol_filter(df[column], 21, 7)
                plot_figure(range(len(df['timestamp'])), initial_data, df[column], column, dir_figures, file[0:-4])
                    
            df.to_csv(dir_smoothed + file , index=False)


