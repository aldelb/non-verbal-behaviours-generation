# Customize the configuration file 
Not all fields are used for all models, they must be filled but will be ignored during training. 
For example, the type of hidden layers can only be customised for the CGAN model.

```
[MODEL_TYPE]
model = id of the model [1: CGAN, 2: AED, 3: ED]
unroll_steps = number of unroll step in case of adversarial learning [False; [integer]]
layer = type of layer, editable only for CGAN [CONV;LSTM;GRU] 
hidden_size = [integer]
first_kernel_size = [odd integer]
kernel_size = [odd integer]
dropout = [number] between 0 and 1

[PATH]
datasets = name of the dataset, usefull for the output names
dir_path = path of the project
data_path = path of the data
saved_path = path to save models and results during learning
output_path = path to save output file (the generated behaviours)
evaluation_path = path to save results of evaluation
model_path = automatically created during training, use during evaluation and/or generation

[TRAIN]
n_epochs = [integer]
batch_size = [integer]
d_lr = discriminator learning rate: [number] between 0 and 1
g_lr = generator learning rate: [number] between 0 and 1
log_interval = The frequency in number of epochs for the backups of training and graphics: [integer]
adversarial_coeff = only for AED, coefficient of the "adversarial" part in the loss function: [number] between 0 and 1

[DATA]
noise_size = CGAN input noise size: [integer]
pose_size = 11
eye_size = 8
pose_t_size = 3
pose_r_size = 3
au_size = 17
derivative = If the first and second derivatives are considered in the input features: [False; True]

[selected_opensmile_columns]
0 = Loudness_sma3
1 = F0semitoneFrom27.5Hz_sma3nz
2 = shimmerLocaldB_sma3nz
3 = logRelF0-H1-H2_sma3nz
4 = logRelF0-H1-A3_sma3nz
5 = mfcc1_sma3
6 = mfcc2_sma3
7 = mfcc3_sma3
8 = mfcc4_sma3

[opensmile_columns]
0 = Loudness_sma3
1 = alphaRatio_sma3
2 = hammarbergIndex_sma3
3 = slope0-500_sma3
4 = slope500-1500_sma3
5 = spectralFlux_sma3
6 = mfcc1_sma3
7 = mfcc2_sma3
8 = mfcc3_sma3
9 = mfcc4_sma3
10 = F0semitoneFrom27.5Hz_sma3nz
11 = jitterLocal_sma3nz
12 = shimmerLocaldB_sma3nz
13 = HNRdBACF_sma3nz
14 = logRelF0-H1-H2_sma3nz
15 = logRelF0-H1-A3_sma3nz
16 = F1frequency_sma3nz
17 = F1bandwidth_sma3nz,
18 = F1amplitudeLogRelF0_sma3nz
19 = F2frequency_sma3nz
20 = F2bandwidth_sma3nz
21 = F2amplitudeLogRelF0_sma3nz
22 = F3frequency_sma3nz
23 = F3bandwidth_sma3nz
24 = F3amplitudeLogRelF0_sma3nz

[openface_columns]
0 = timestamp
1 = gaze_0_x
2 = gaze_0_y
3 = gaze_0_z
4 = gaze_1_x
5 = gaze_1_y
6 = gaze_1_z
7 = gaze_angle_x
8 = gaze_angle_y
9 = pose_Tx
10 = pose_Ty
11 = pose_Tz
12 = pose_Rx
13 = pose_Ry
14 = pose_Rz
15 = AU01_r
16 = AU02_r
17 = AU04_r
18 = AU05_r
19 = AU06_r
20 = AU07_r
21 = AU09_r
22 = AU10_r
23 = AU12_r
24 = AU14_r
25 = AU15_r
26 = AU17_r
27 = AU20_r
28 = AU23_r
29 = AU25_r
30 = AU26_r
31 = AU45_r


```
