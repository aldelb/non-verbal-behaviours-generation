import argparse
from genericpath import isdir
import os
import numpy as np
import torch
import constants.constants as constants
from constants.constants_utils import read_params
from utils.create_final_file import createFinalFile
from utils.model_utils import find_model, load_model
from torch_dataset import TestSet
import pandas as pd

gaze_columns = ["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y"]
translation_columns = ["pose_Tx", "pose_Ty", "pose_Tz"]
au_columns = ["pose_Rx", "pose_Ry", "pose_Rz", "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-params', help='Path to the constant file', default="./params.cfg")
    parser.add_argument('-epoch', help='number of epoch of recovred model', default=9)
    parser.add_argument('-video', help='wich video we want to generate', default="all")
    parser.add_argument('-dataset', help='wich video we want to generate', default="")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parser.parse_args()
    read_params(args.params, "generate")

    if(args.dataset != ""):
        datasets = args.dataset
        constants.datasets = datasets.split(",")

    model_file = find_model(int(args.epoch)) 
    model = load_model(constants.dir_path + constants.saved_path + model_file, device)

    path_data_out = constants.dir_path + constants.output_path + model_file[0:-3] + "/"
    if(not isdir(path_data_out)):
        os.makedirs(path_data_out, exist_ok=True)

    test_set = TestSet()

    gened_seqs = []
    columns = constants.openface_columns

    current_part = 0 
    current_key = ""
    df_list = []

    for index, data in enumerate(test_set, 0):
        input, target = data[0], data[1]
        if(constants.model_number in [7,8,9]):
            input = target
        key = test_set.getInterval(index)[0]
        if(args.video == "all" or key == args.video + "[0]"):
            if(current_key != key): #process of a new video
                if(current_key != ""):
                    createFinalFile(path_data_out, current_key, df_list)
                print("Generation of video", key[0:-3] , "...")
                current_part = 0
                current_key = key
                df_list = []

            if(constants.prosody_size == 1):
                input = np.reshape(input, input.shape + (1,))

            if(constants.model_number in [10, 11, 12]):
                out = constants.generate_motion(model, input, target)
            else:
                out = constants.generate_motion(model, input)
            #add timestamp and head translation for greta
            timestamp = np.array(test_set.getInterval(index)[1][:,0])
            out = np.concatenate((timestamp.reshape(-1,1), out[:,:constants.eye_size], np.zeros((out.shape[0], 3)), out[:,constants.eye_size:]), axis=1)
            df = pd.DataFrame(data = out, columns = columns)
            df_list.append(df)
            current_part += 1

    createFinalFile(path_data_out, current_key, df_list)