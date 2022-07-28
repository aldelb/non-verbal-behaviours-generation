import argparse
from genericpath import isdir
import os
import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import constants.constants as constants
from constants.constants_utils import read_params
from torch_dataset import TestSet
from utils.model_utils import find_model

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def getData(path_data_out):
    test_set = TestSet()
    gened_seqs = []
    for file in os.listdir(path_data_out):
        pd_file = pd.read_csv(path_data_out + file)
        pd_file = pd_file[["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y", "pose_Rx", "pose_Ry",
                "pose_Rz", "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]]
        gened_seqs.append(pd_file)

    gened_frames = np.concatenate(gened_seqs, axis=0)
    real_frames = np.concatenate(test_set.Y_final_ori, axis=0)

    return real_frames, gened_frames


def create_pca(real_frames, gened_frames, path_evaluation):
    scaler = StandardScaler()
    scaler.fit(real_frames)
    X_gened = scaler.transform(gened_frames) 
    X_real = scaler.transform(real_frames) 

    mypca = PCA(n_components=2, random_state = 1) # On paramètre ici pour ne garder que 2 axes
    mypca.fit(X_real)

    print(mypca.singular_values_) # Valeurs de variance
    print('Explained variation per principal component: {}'.format(mypca.explained_variance_ratio_))
    data_generated = mypca.transform(X_gened)
    data_real = mypca.transform(X_real)

    df_generated = pd.DataFrame(data = data_generated, columns = ['principal component 1', 'principal component 2'])
    df_real = pd.DataFrame(data = data_real, columns = ['principal component 1', 'principal component 2'])

    indicesToKeep = random.sample(range(len(df_generated)), 1000)

    fig = plt.figure(dpi=100)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.scatter(df_real.loc[indicesToKeep, 'principal component 1'], df_real.loc[indicesToKeep, 'principal component 2'], label='Real data')
    ax1.scatter(df_generated.loc[indicesToKeep, 'principal component 1'], df_generated.loc[indicesToKeep, 'principal component 2'], label='Generated data')
    ax1.set_xlabel('Principal Component - 1')
    ax1.set_ylabel('Principal Component - 2')
    ax1.legend()
    plt.savefig(path_evaluation + 'pca.png')
    plt.close()
    
def calculate_kde(real_frames, gened_frames, path_evaluation, bandwidth = None):
    if(bandwidth == None):
        params = {'bandwidth':  np.logspace(-2, 0, 5)}
        print("Grid search for bandwith parameter of Kernel Density...")
        grid = GridSearchCV(KernelDensity(kernel='gaussian'), params, cv=3)
        grid.fit(gened_frames)
        print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
        scores = grid.best_estimator_.score_samples(real_frames)

    else:
        kde = KernelDensity(kernel='gaussian', bandwidth = float(bandwidth)).fit(gened_frames)
        scores = kde.score_samples(real_frames)

    mean = np.mean(scores)
    sd = np.std(scores)
    print("mean ", str(mean))
    print("ses ", str(sd))

    f = open(path_evaluation + "eval.txt", "w")
    f.write("mean "+ str(mean) + "\n")
    f.write("sd "+ str(sd) + "\n")
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-params', help='Path to the constant file', default="./params.cfg")
    parser.add_argument('-epoch', help='number of epoch of recovred model', default=9)
    parser.add_argument('-bandwidth', help= "bandwith in known", default=None)
    args = parser.parse_args()
    read_params(args.params, "eval")

    model_file = find_model(int(args.epoch)) 

    path_data_out = constants.dir_path + constants.output_path + model_file[0:-3] + "/"
    if(not isdir(path_data_out)):
        raise Exception(path_data_out + "is not a directory")

    path_evaluation = constants.dir_path + constants.evaluation_path + model_file[0:-3] + "/"
    if(not isdir(path_evaluation)):
        os.makedirs(path_evaluation, exist_ok=True)

    real_frames, gened_frames = getData(path_data_out)

    print("create pca...")
    create_pca(real_frames, gened_frames, path_evaluation)
    print("calculate kde...")
    calculate_kde(real_frames, gened_frames, path_evaluation, args.bandwidth)
    
