# --- type params
model_number = 0
unroll_steps =  False
layer = ""
hidden_size = 0
first_kernel_size = 0
kernel_size = 0
dropout = 1

# --- Path params
datasets = ""
dir_path = ""
data_path = ""
saved_path = ""
output_path = ""
evaluation_path = ""
model_path = ""

# --- Training params
n_epochs =  0
batch_size = 0
d_lr =  0
g_lr =  0
log_interval =  0
adversarial_coeff = 0


# --- Data params
noise_size = 0
pose_size = 0 # nombre de colonne openface pose and gaze angle
eye_size = 0 #nombre de colonne openface gaze (déja normalisé)
pose_t_size = 0 #location of the head with respect to camera
pose_r_size = 0 # Rotation is in radians around X,Y,Z axes with camera being the origin.
au_size = 0 # nombre de colonne openface AUs
prosody_size = 0 #nombre de colonne opensmile selectionnées
derivative = False

opensmile_columns = []
selected_opensmile_columns = []
selected_os_index_columns = []
openface_columns = []

#Model function
model = None
train_model = None
generate_motion = None

