import os
import sys
import numpy as np
import pandas as pd
import numpy as np
import pickle
from genericpath import isfile

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import mmsdk
from mmsdk import mmdatasdk as md
from mmsdk.mmdatasdk.dataset.standard_datasets.CMU_MOSI import cmu_mosi_std_folds as standard_folds_mosi

#Align visual and audio features on visual features 
audio_features = ['Loudness_sma3','alphaRatio_sma3','hammarbergIndex_sma3','slope0-500_sma3','slope500-1500_sma3','spectralFlux_sma3','mfcc1_sma3','mfcc2_sma3','mfcc3_sma3','mfcc4_sma3','F0semitoneFrom27.5Hz_sma3nz','jitterLocal_sma3nz','shimmerLocaldB_sma3nz','HNRdBACF_sma3nz','logRelF0-H1-H2_sma3nz','logRelF0-H1-A3_sma3nz','F1frequency_sma3nz','F1bandwidth_sma3nz','F1amplitudeLogRelF0_sma3nz','F2frequency_sma3nz','F2bandwidth_sma3nz','F2amplitudeLogRelF0_sma3nz','F3frequency_sma3nz','F3bandwidth_sma3nz','F3amplitudeLogRelF0_sma3nz']

visual_features = ["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y", "pose_Rx", "pose_Ry",
                "pose_Rz", "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]

label_features = ['label']

def init_audio_data(compseq, audio_processed_path, video_to_transform):
    compseq = {}
    duration = {}
    for vid_key in video_to_transform:
        if(isfile(audio_processed_path + vid_key + '.csv')):
            compseq[vid_key] = {}
            dataset = pd.read_csv(audio_processed_path + vid_key + '.csv')
            intervals = dataset[['start', 'end']]
            intervals['start'] = intervals['start'].transform(lambda x :  pd.Timedelta(x).total_seconds())
            intervals['end'] = intervals['end'].transform(lambda x :  pd.Timedelta(x).total_seconds())	
            compseq[vid_key]["intervals"] = intervals.to_numpy()
            duration[vid_key] = intervals.to_numpy()[-1][1]
            compseq[vid_key]["features"] = dataset[audio_features].to_numpy()

    return compseq, duration

def init_visual_data(compseq, duration, visual_processed_path, video_to_transform):
    compseq = {}
    for vid_key in video_to_transform:
        if(isfile(visual_processed_path + vid_key + '.csv')):
            compseq[vid_key] = {}
            dataset = pd.read_csv(visual_processed_path + vid_key + '.csv')
            intervals = dataset[['timestamp']]
            last_raw = {'timestamp' : duration[vid_key]}
            compseq[vid_key]["intervals"] = np.append(intervals, dataset[['timestamp']].iloc[1: , :].append(last_raw, ignore_index=True), axis = 1)
            dataset = dataset[visual_features]
            compseq[vid_key]["features"] = dataset.to_numpy()
    return compseq

def init_label_data(compseq, duration, visual_processed_path, video_to_transform):
    compseq = {}
    for vid_key in video_to_transform:
        if(isfile(visual_processed_path + vid_key + '.csv')):
            compseq[vid_key] = {}
            label = [[0.5]]
            intervals = [[0.0, duration[vid_key]]]
            compseq[vid_key]["intervals"] = np.array(intervals)
            compseq[vid_key]["features"] = np.array(label)
    return compseq


def create_audio_csd(audio_path, audio_csd_path, video_to_transform):
    audio_data = {}
    duration = {}
    audio_data, duration = init_audio_data(audio_data, audio_path, video_to_transform)
    audio_comput = md.computational_sequence("audio_data")
    audio_comput.setData(audio_data,"audio_data")
    metadata1 = {
            "root name" : "OpenSmile data",
            "computational sequence description" :"prosodic features extracted with openSmile",
            "dimension names" : audio_features,
            "computational sequence version" : "",
            "alignment compatible" : "",
            "dataset name" : "",
            "dataset version" : "",
            "creator" : "",
            "contact" : "",
            "featureset bib citation" : "",
            "dataset bib citation" : ""
            }
    audio_comput.set_metadata(metadata1)
    audio_comput.deploy(audio_csd_path)
    return duration


def create_visual_csd(duration, visual_path, visual_csd_path, video_to_transform):
    visual_data = {}
    visual_data = init_visual_data(visual_data, duration, visual_path, video_to_transform)
    visual_comput = md.computational_sequence("visual_data")
    visual_comput.setData(visual_data,"visual_data")

    metadata = {
            "root name" : "OpenFace Data",
            "computational sequence description" :"visual features extracted with openSmile",
            "dimension names" : visual_features,
            "computational sequence version" : "",
            "alignment compatible" : "",
            "dataset name" : "",
            "dataset version" : "",
            "creator" : "",
            "contact" : "",
            "featureset bib citation" : "",
            "dataset bib citation" : ""
            }
    visual_comput.set_metadata(metadata)
    visual_comput.deploy(visual_csd_path)

def create_label_csd(duration, visual_path, label_csd_path, video_to_transform):
    label_data = {}
    label_data = init_label_data(label_data, duration, visual_path, video_to_transform)
    label_comput = md.computational_sequence("label_data")
    label_comput.setData(label_data,"label_data")

    metadata = {
            "root name" : "Attitude label",
            "computational sequence description" :"label to create with plateform",
            "dimension names" : label_features,
            "computational sequence version" : "",
            "alignment compatible" : "",
            "dataset name" : "",
            "dataset version" : "",
            "creator" : "",
            "contact" : "",
            "featureset bib citation" : "",
            "dataset bib citation" : ""
            }
    label_comput.set_metadata(metadata)
    label_comput.deploy(label_csd_path)

def create_align_dataset(label_csd_path, audio_csd_path, visual_csd_path, align_csd_path):
    # align on visual features and on labels
    mydataset_recipe = {"audio":audio_csd_path,"visual":visual_csd_path}
    mydataset = md.mmdataset(mydataset_recipe)

    # we define a simple averaging function that does not depend on intervals
    def avg(intervals: np.array, features: np.array) -> np.array:
        try:
            return np.average(features, axis=0)
        except:
            return features

    mydataset.align("visual", collapse_functions=[avg])
    label_recipe = {"label": label_csd_path}
    mydataset.add_computational_sequences(label_recipe, destination=None)
    mydataset.align("label")
    with open(align_csd_path, 'wb') as f:
        pickle.dump(mydataset, f)
    return mydataset

def create_set(name, mydataset, output_path, collection, type_of_set):
    X = []
    Y = []
    intervals = []
    for vid_key in collection:
        vid_key = vid_key + '[0]'
        if(vid_key in mydataset['visual'].keys()):
            segment_len = 120
            overlap = 10
            i_max = len(mydataset['visual'][vid_key]['features'])
            #toutes les 4 sec, avec recouvrement 10 frames
            i = 0
            while(i+segment_len < i_max):
                print(np.array(mydataset['audio'][vid_key]['features'][i:i+segment_len]).shape)
                X.append(mydataset['audio'][vid_key]['features'][i:i+segment_len]) 
                Y.append(mydataset['visual'][vid_key]['features'][i:i+segment_len])
                intervals.append([vid_key, mydataset['audio'][vid_key]['intervals'][i:i+segment_len]])
                i+= segment_len - overlap
    
    with open(output_path + "X_"+type_of_set+"_"+name+".p", 'wb') as f:
        pickle.dump(X, f)
    with open(output_path + "y_"+type_of_set+"_"+name+".p", 'wb') as f:
        pickle.dump(Y, f)
    with open(output_path + "intervals_"+type_of_set+"_"+name+".p", 'wb') as f:
        pickle.dump(intervals, f)

def create_final_y_test(test_key, processed_video_path, output_path, dataset_name):
        output_path = output_path + "y_test_final_"+dataset_name+".p"
        all_df = []
        print(np.array(test_key).shape)
        for file in os.listdir(processed_video_path):
            if(file[-4:] == ".csv" and file[0:-4] in test_key):
                df = pd.read_csv(processed_video_path + file)
                df = df[["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y", "pose_Rx", "pose_Ry",
                "pose_Rz", "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]]
                all_df.append(df)
        print(np.array(all_df).shape)
        with open(output_path, 'wb') as f:
            pickle.dump(all_df, f)


def main():
    dataset_name = "mosi"
    standard_fold = standard_folds_mosi
    dataset_path = "./data/"+dataset_name+"_data/"

    output_path = dataset_path + "Features/"
    audio_path = dataset_path + "Raw/Audio/processed/"
    visual_path = dataset_path + "Raw/Video/processed/"
    audio_csd_path = dataset_path + "Raw/Audio/audio.csd"
    visual_csd_path = dataset_path + "Raw/Video/visual.csd"
    label_csd_path = dataset_path + "Raw/label.csd"
    align_csd_path = dataset_path + "Raw/align.csd"

    print("*"*10, "beginning", "*"*10)
    duration = {}
    videos_to_transform = standard_fold.standard_train_fold + standard_fold.standard_valid_fold + standard_fold.standard_test_fold

    # duration = create_audio_csd(audio_path, audio_csd_path, videos_to_transform)
    # print("*"*10, "audio csd created", "*"*10)

    # create_visual_csd(duration, visual_path, visual_csd_path, videos_to_transform)
    # print("*"*10, "visual csd created", "*"*10)

    # create_label_csd(duration, visual_path, label_csd_path, videos_to_transform)
    # print("*"*10, "label csd created", "*"*10)

    dataset = create_align_dataset(label_csd_path, audio_csd_path, visual_csd_path, align_csd_path)
    print("*"*10, "visual and audio features aligned", "*"*10)

    create_set(dataset_name, dataset, output_path, standard_fold.standard_train_fold + standard_fold.standard_valid_fold, "train")
    print("*"*10, "train set created", "*"*10)

    create_set(dataset_name, dataset, output_path, standard_fold.standard_test_fold, "test")
    print("*"*10, "test set created", "*"*10)

    create_final_y_test(standard_fold.standard_test_fold, visual_path, output_path, dataset_name)
    print("*"*10, "final test file created", "*"*10)
    print("*"*10, "end", "*"*10)

    return 0

if __name__ == "__main__":
    sys.exit(main())
