import constants.constants as constants

def write_params(f, title, params):
    f.write(f"# --- {title}\n")
    for argument in params.keys() :
        f.write(f"{argument} : {params[argument]}\n\n")

def save_params(saved_path, model, D = None):
    path_params = {"saved path" : saved_path}
    training_params = {
        "n_epochs" : constants.n_epochs,
        "batch_size" : constants.batch_size,
        "d_lr" : constants.d_lr,
        "g_lr" : constants.g_lr,
        "adversarial_coeff" : constants.adversarial_coeff}

    model_params = {
        "model" : constants.model_number,
        "unroll_steps" : constants.unroll_steps,
        "layer" : constants.layer,
        "hidden_size" : constants.hidden_size,
        "first_kernel_size" : constants.first_kernel_size,
        "kernel_size" : constants.kernel_size,
        "dropout" : constants.dropout}

    data_params = {
        "log_interval" : constants.log_interval,
        "noise_size" : constants.noise_size,
        "prosody_size" : constants.prosody_size,
        "pose_size" : constants.pose_size,
        "au_size" : constants.au_size,
        "column keep in opensmile" : constants.selected_opensmile_columns,
        "derivative" : constants.derivative}

    file_path = saved_path + "parameters.txt"
    f = open(file_path, "w")
    write_params(f, "Model params", model_params)
    write_params(f, "Path params", path_params)
    write_params(f, "Training params", training_params)
    write_params(f, "Data params", data_params)

    f.write("-"*10 + "Models" + "-"*10 + "\n")
    f.write("-"*10 + "Generateur" + "-"*10 + "\n")
    f.write(str(model))
    if(D != None):
        f.write("-"*10 + "Discriminateur" + "-"*10 + "\n")
        f.write(str(D))
    f.close()