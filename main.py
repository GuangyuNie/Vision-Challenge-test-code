from fmodel import create_model
from adversarial_vision_challenge import model_server


if __name__ == '__main__':
    fmodel = create_model()
    model_server(fmodel)
