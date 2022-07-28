import os
from os import listdir
from os.path import isfile, join
import sys
import subprocess
import opensmile

dataset_name = "mosi"
path = "data/"+dataset_name+"_data/Raw/Audio/"
audio_dir = "WAV_16000/Full/"


check_only = sys.argv[1] ## check the number of files that still need to be processed 
dir = path + audio_dir
out = path + "processed/"
os.mkdir(out)

to_create = []
number_of_video = len(listdir(dir))

for f in listdir(dir):
    if(".wav" in f):
        csv_file = f[0:-4] + ".csv"
        if not isfile(join(out, csv_file)):
            print(join(dir,f))
            to_create.append(csv_file)
            if not (check_only == 'true'):
                smile = opensmile.Smile(
                        feature_set=opensmile.FeatureSet.eGeMAPSv02,
                        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                        )
                result_smile = smile.process_file(join(dir,f))
                result_smile.to_csv(join(out, csv_file), sep=',')
        
if(check_only == 'true'):
    print("to be processed ", len(to_create), "/", number_of_video)
