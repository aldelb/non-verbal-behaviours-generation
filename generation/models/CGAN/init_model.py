import constants.constants as constants
from models.CGAN.generating import GenerateModel1
from models.CGAN.model import Generator as model1
from models.CGAN.training import TrainModel1

def init_model_1(task):
    if(task == "train"):
        train = TrainModel1(gan=True)
        constants.train_model = train.train_model
    elif(task == "generate"):
        constants.model = model1
        generator = GenerateModel1()
        constants.generate_motion = generator.generate_motion