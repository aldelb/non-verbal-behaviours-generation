import torch
import constants.constants as constants
from models.GenerateClass import Generate
from utils.noise_generator import NoiseGenerator

class GenerateModel1(Generate):
    def __init__(self):
        super(GenerateModel1, self).__init__()

    def generate_motion(self, model, prosody):
        prosody = self.reshape_prosody(prosody)

        noise_g = NoiseGenerator()
        noise = noise_g.gaussian_variating(T=prosody.shape[0], F=40, size=constants.noise_size, allow_indentical=True)
        noise = torch.FloatTensor(noise).unsqueeze(1).to(self.device)

        with torch.no_grad():
            outs = model.forward(prosody, noise)
        outs = self.reshape_single_output(outs)
        return outs

