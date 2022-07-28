import os
from os import listdir
from os.path import isfile, join
import sys
import subprocess

dataset_name = "mosi"
openFace_dir = "PATH/TO/OPENFACE/PROJECT"
path = "data/"+dataset_name+"_data/Raw/Video/"
video_dir = "Full/"

check_only = sys.argv[1] ## check the number of files that still need to be processed 

dir = path + video_dir
out = path + "processed/"
os.mkdir(out)

to_create = []
number_of_video = len(listdir(dir))

for f in listdir(dir):
    print(join(dir,f))
    csv_file = f[0:-4] + ".csv"
    if not isfile(join(out, csv_file)):
        print(csv_file)
        to_create.append(csv_file)
    if isfile(join(dir, f)) and not isfile(join(out, csv_file)) and not (check_only == 'true'):
        subprocess.Popen(openFace_dir + "/build/bin/FaceLandmarkVidMulti -f" + join(dir, f) + '-out_dir '+ out, shell=True).wait()
if(check_only == 'true'):
    print("to be processed ", len(to_create), "/", number_of_video)
