import configparser
import constants.constants as constants
import os
from datetime import date

config = configparser.RawConfigParser()

def read_params(file, task, id=None):

    config.read(file)

    # --- model type params
    constants.model_number =  config.getint('MODEL_TYPE','model')
    try :
        constants.unroll_steps =  config.getboolean('MODEL_TYPE','unroll_steps')
    except:
        constants.unroll_steps =  config.getint('MODEL_TYPE','unroll_steps')
    constants.layer =  config.get('MODEL_TYPE','layer')
    constants.hidden_size =  config.getint('MODEL_TYPE','hidden_size') 
    constants.kernel_size = config.getint('MODEL_TYPE','kernel_size') 
    constants.first_kernel_size = config.getint('MODEL_TYPE','first_kernel_size') 
    constants.dropout =  config.getfloat('MODEL_TYPE','dropout') 

    datasets = config.get('PATH','datasets')
    constants.datasets = datasets.split(",")
    constants.dir_path = config.get('PATH','dir_path')
    constants.data_path = config.get('PATH','data_path')
    constants.saved_path = config.get('PATH','saved_path')
    constants.output_path = config.get('PATH','output_path')
    constants.evaluation_path = config.get('PATH','evaluation_path')
    constants.model_path = config.get('PATH','model_path')

    # --- Training params
    constants.n_epochs =  config.getint('TRAIN','n_epochs')
    constants.batch_size = config.getint('TRAIN','batch_size')
    constants.d_lr =  config.getfloat('TRAIN','d_lr')
    constants.g_lr =  config.getfloat('TRAIN','g_lr')
    constants.log_interval =  config.getint('TRAIN','log_interval')
    constants.adversarial_coeff = config.getfloat('TRAIN','adversarial_coeff')

    # --- Data params
    constants.noise_size = config.getint('DATA','noise_size')
    constants.pose_size = config.getint('DATA','pose_size') 
    constants.eye_size = config.getint('DATA','eye_size')
    constants.pose_t_size = config.getint('DATA','pose_t_size')
    constants.pose_r_size = config.getint('DATA','pose_r_size')
    constants.au_size = config.getint('DATA','au_size') 
    constants.derivative = config.getboolean('DATA','derivative')

    constants.opensmile_columns = get_config_columns('opensmile_columns')
    constants.selected_opensmile_columns = get_config_columns('selected_opensmile_columns')
    constants.openface_columns = get_config_columns('openface_columns')

    base_size = len(constants.selected_opensmile_columns) 

    if constants.derivative:
        constants.prosody_size = base_size * 3
    else:
        constants.prosody_size = base_size

    constants.selected_os_index_columns = []
    for column in constants.selected_opensmile_columns:
        constants.selected_os_index_columns.append(constants.opensmile_columns.index(column))

    if(task == "train"):
        constants.saved_path = create_saved_path(file, id)
    set_model_function(constants.model_number, task)

def set_model_function(model_number, task):
    if(model_number == 1): #CGAN
        from models.CGAN.init_model import init_model_1
        init_model_1(task)
    elif(model_number == 2): #AED
        from models.AED.init_model import init_model_2
        init_model_2(task)
    elif(model_number == 3): #ED (multiple decoders)
        from models.ED.init_model import init_model_3
        init_model_3(task)
    else:
        raise Exception("Model ", model_number, " does not exist")

def get_config_columns(list):
    list_items = config.items(list)
    columns = []
    for _, column_name in list_items:
        columns.append(column_name)
    return columns

def create_saved_path(config_file, id):
    # * Create dir for store models, hist and params
    today = date.today().strftime("%d-%m-%Y")
    saved_path = constants.dir_path + constants.saved_path 
    str_dataset = ""
    for dataset in constants.datasets:
        str_dataset += dataset + "_"

    dir_path = f"{today}_{str_dataset}"
    if(id == "0"):
        i = 1
        while(os.path.isdir(saved_path + dir_path + f"{i}")):
            i = i+1
        dir_path += f"{i}/"
    else:
        dir_path += f"{id}/"
    saved_path += dir_path
    os.makedirs(saved_path, exist_ok=True)
    config.set('PATH','model_path', dir_path)
    with open(config_file, 'w') as configfile:
        config.write(configfile)
    return saved_path