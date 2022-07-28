import argparse
import constants.constants as constants
from constants.constants_utils import read_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-params', help='Path to the constant file', default="./params.cfg")
    parser.add_argument('-id', help='Path to save result and models', default="0")
    args = parser.parse_args()

    read_params(args.params, "train", args.id)
    constants.train_model()
    