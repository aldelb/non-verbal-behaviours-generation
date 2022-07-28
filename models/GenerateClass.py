import torch
import constants.constants as constants
from torch_dataset import TrainSet

class Generate():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("*"*10, "cuda available ", torch.cuda.is_available(), "*"*10)
        self.dset = TrainSet()

    def reshape_prosody(self, input):
        input = self.dset.scale_x(input)
        input = torch.tensor(input, device = self.device).unsqueeze(0).float() #batch size 1
        input = torch.reshape(input, (-1, input.shape[2], input.shape[1]))
        return input
    
    def reshape_pose(self, input):
        input = self.dset.scale_y(input)
        input = torch.tensor(input, device = self.device).unsqueeze(0).float() #batch size 1
        input = torch.reshape(input, (-1, input.shape[2], input.shape[1]))
        return input

    def separate_openface_features(self, input):
        input = torch.reshape(input, (-1, input.shape[2], input.shape[1]))
        
        input_eye = torch.index_select(input, dim=2, index = torch.tensor(range(constants.eye_size), device=self.device))
        input_eye = torch.reshape(input_eye, (-1, input_eye.shape[2], input_eye.shape[1]))
        
        input_pose_r = torch.index_select(input, dim=2, index=torch.tensor(range(constants.eye_size, constants.eye_size + constants.pose_r_size), device=self.device))
        input_pose_r = torch.reshape(input_pose_r, (-1, input_pose_r.shape[2], input_pose_r.shape[1]))
        
        input_au = torch.index_select(input, dim=2, index=torch.tensor(range(constants.pose_size, constants.pose_size + constants.au_size), device=self.device))
        input_au = torch.reshape(input_au, (-1, input_au.shape[2], input_au.shape[1]))

        return input_eye, input_pose_r, input_au
    

    def reshape_output(self, output_eye, output_pose_r, output_au):
        output_eye = torch.reshape(output_eye, (-1, output_eye.shape[2], output_eye.shape[1]))
        output_pose_r = torch.reshape(output_pose_r, (-1, output_pose_r.shape[2], output_pose_r.shape[1]))
        output_au = torch.reshape(output_au, (-1, output_au.shape[2], output_au.shape[1]))
        outs = torch.cat((output_eye, output_pose_r, output_au), 2)
        outs = torch.squeeze(outs)
        outs = torch.reshape(outs, (-1, outs.shape[1]))
        outs = self.dset.rescale_y(outs.cpu())
        return outs
    
    def reshape_single_output(self, outs):
        outs = torch.reshape(outs, (-1, outs.shape[1]))
        outs = self.dset.rescale_y(outs.cpu())
        return outs