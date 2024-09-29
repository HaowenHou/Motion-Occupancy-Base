from models.base_motion_transf import BaseMotionTransf
from models.ctrl_transf import CtrlTransf


model_dict = {
    'BASE_MOTION_TRANSF': BaseMotionTransf,
    'CTRL_TRANSF': CtrlTransf,
}

def get_model(config):
    return model_dict[config.MODEL.NAME](config)
