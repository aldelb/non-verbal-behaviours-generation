import torch
import torch.nn as nn
import constants.constants as constants

def find_model(epoch):
    model_file = constants.model_path + "epoch"
    model_file += f"_{epoch}.pt"
    return model_file

def load_model(param_path, device):
    model = constants.model()
    model.load_state_dict(torch.load(param_path, map_location=device))
    return model.to(device)

def saveModel(model, epoch, saved_path):
    torch.save(model.state_dict(), f'{saved_path}epoch_{epoch}.pt')
