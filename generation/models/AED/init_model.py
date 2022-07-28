import constants.constants as constants
from models.AED.generating import GenerateModel2
from models.AED.model import Generator as model2
from models.AED.training import TrainModel2


def init_model_2(task):
    if(task == "train"):
        train = TrainModel4(gan=True)
        constants.train_model = train.train_model
    elif(task == "generate"):
        constants.model = model2
        generator = GenerateModel2()
        constants.generate_motion = generator.generate_motion