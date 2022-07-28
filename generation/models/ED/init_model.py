import constants.constants as constants
from models.ED.generating import GenerateModel3
from models.ED.training import TrainModel3
from models.ED.model import AutoEncoder as model3


def init_model_3(task):
    if(task == "train"):
        train = TrainModel3(gan=False)
        constants.train_model = train.train_model
    else:
        constants.model = model3
        generator = GenerateModel3()
        constants.generate_motion = generator.generate_motion