# Automatic facial expressions, gaze direction and head movements generation of a virtual agent

The code contains two models to jointly and automatically generate the head, facial and gaze movements (non-verbal behaviours) of a virtual agent from acoustic speech features. Two architectures are explored: a Generative Adversarial Network and an Adversarial Encoder-Decoder. Head movements and gaze orientation are generated as 3D coordinates, while facial expressions are generated using action units based on the facial action coding system. 

## Example

![A simple sample](docs/example.gif)

The sound is not added to this gif. To see more examples go [here](https://www.youtube.com/channel/UCCds0WJg3qbwYtUKSjKJqzw/featured) (there are videos animated with ground truth, CGAN model, AED model, ED model).

## To reproduce
1. Clone the repository
2. In a conda console, execute 'conda env create -f environment.yml' to create the right conda environment. Go to the project location.

### Data and features extraction 
We extract the speech and visual features automatically from these videos using existing tools. You can also directly recover the extracted and align features with the next section.

1. Download the CMU_MOSI dataset from http://immortal.multicomp.cs.cmu.edu/raw_datasets/
2. Create a directory './data/mosi_data' and inside it, a directory "Features". Put the downloaded zip into "data", then unzip it. You must obtain the following structure :
```
data
-----mosi_data
--------Features
--------Raw
---------------Audio
---------------Transcript
---------------Video
```
3. Download and install OpenFace using : https://github.com/TadasBaltrusaitis/OpenFace.git (the installation depends on your system). 
4. In the file "pre-processing/extract_openface.py" change "openFace_dir = "PATH/TO/OPENFACE/PROJECT" and put the absolute path of the project.
5. In the conda console, extract the speech features by executing "python pre_processing/extract_opensmile.py false".
6. In the conda console, extract the visual features by executing "python pre_processing/extract_openface.py false".
7. In the conda console, align the speech and visual modalities by executing "python pre_processing/align_data_mmsdk.py".

### Features recovery
You can also directly recover the extracted and align features.

1. Create a directory './data/mosi_data' and inside it, a directory "Features".
2. Download files found in [this drive](https://drive.google.com/drive/folders/1ZEV_I7qQTPlKRULAZ90Nf6P9C7yU6rQZ?usp=sharing) and place them in the repository "Features".

### Models training
1. In the directory "generation", you will find "params.cfg". 
It is the configuration file to customise the model before training. 
To learn what section needs to be change go see [the configuration file](docs/config_file.md).
2. You can conserve the existing file or create a new one. 
3. In the conda console, train the model by executing "python train.py -params PATH/TO/CONFIG/FILE.cfg [-id NAME_OF_MODEL]"
You can visualise the created graphics during training in the repository "generation/saved_models" or the one put in "saved_path" in the configuration file. 

### Behaviours generation
1. In the conda console, generate behaviours by executing "python generation/generate.py -epoch [integer] -params PATH/TO/CONFIG/FILE.cfg -dataset mosi". The behaviours are generated in the form of 3D coordinates and intensity of facial action units. These are csv files stored in the directory "generation/data/output/MODEL_PATH" or the one put in "output_path" in the configuration file.

- -epoch : during training, if you trained in 100 epochs, recording every 10 epochs, you must enter a number within [10;20;30;40;50;60;70;80;90;99].
- -params : path to the config file. 
- -dataset : name of the considered dataset. 

### Models evaluation
The objective evaluation of these models is conducted with measures such as density evaluation
and a visualisation from PCA reduction.

1.  In the conda console, evaluate model objectively by executing "python generation/evaluate.py -params PATH/TO/CONFIG/FILE.cfg -epoch [integer] -bandwidth 0.1"

- -epoch : during training, if you trained on 100 epochs, recording every 10 epochs, you must enter a number within [10;20;30;40;50;60;70;80;90;99]. You must perform the generation before the evaluation. 
- -params : path to the config file. 
- -bandwith : parameter bandwidth for the kernel density estimation. 

You will find the results in the directory "generation/evaluation" or the one put in "evaluation_path" in the configuration file.

### Data post-processing (smoothing)
1. To smooth the generated behaviours, open the file "post_processing/smoothing.py" and change the variable "dirs" by the name of the directory that contains the generated behaviours in csv files. For example "generation/data/output/27-07-2022_mosi_2/epoch_9/".

2. In the conda console, to smooth data execute "python post_processing/smoothing.py false". "true" if you want to visualise the effect of smoothing in graph, "false" otherwise. 

### Animate the generated behaviours
To animate a virtual agent with the generated behaviours, we use the GRETA platform. 

1. Download and install GRETA with "gpl-grimaldi-release.7z" at https://github.com/isir/greta/releases/tag/v1.0.1.
2. Open GRETA. Open the configuration "Greta - Record AU.xml" already present GRETA. 
3. Use the block "AU Parser File Reader" and "Parser Capture Controller AU" to create the video from the csv file generated. 

### Add synthesised voice 
You can directly concatenate the voices from the original videos with ffmpeg as shown in example 1, or change the pitch as shown in example 2. The audio files are in "data/mosi_data/Raw/Audio/WAV_16000/Full"

Example 1 : 
```
ffmpeg -i WAV_FILE.wav -i VIDEO_FILE.avi -c copy -map 1:v:0 -map 0:a:0 OUTPUT.avi
```

Example 2 : 
Change the pitch with python : 
```
rate, data = wavfile.read(WAV_FILE.wav)
# perform noise reduction
reduced_noise = nr.reduce_noise(y=data, sr=rate)
wavfile.write(WAV_FILE_REDUCED_NOISE.wav, rate, reduced_noise)
    
y, sr = librosa.load(WAV_FILE_REDUCED_NOISE.wav, sr=16000) # y is a numpy array of the wav file, sr = sample rate
y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=-1, bins_per_octave=25)   wavfile.write(WAV_FILE_PITCH_CHANGE.wav, sr, y_shifted)
```

Concatenate the created sound and the video:
```
ffmpeg -i WAV_FILE_PITCH_CHANGE.wav -i VIDEO_FILE.avi -c copy -map 1:v:0 -map 0:a:0 OUTPUT.avi
```
