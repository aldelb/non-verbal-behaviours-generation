import torch
from models.GenerateClass import Generate

class GenerateModel3(Generate):
    def __init__(self):
        super(GenerateModel3, self).__init__()

    def generate_motion(self, model, prosody):
        prosody = self.reshape_prosody(prosody)
        with torch.no_grad():
            output_eye, output_pose_r, output_au = model.forward(prosody)
        outs = self.reshape_output(output_eye, output_pose_r, output_au)
        return outs