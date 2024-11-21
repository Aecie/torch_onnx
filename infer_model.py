"""
tensorrt workflow:
initialize:
    logger: 
    builder:
    network construction/load
    serialize:
runtime:
    generate trt inner representation:
    build engine:
    create context:
    bind input/output
    buffer preparation:
    execution:
"""
import onnx
import pycuda
import tensorrt as trt
from model_configs.train_infer import config


infer_config = config['AlexNet_MNIST']

onnx_model = onnx.load_model(config['onnx_model_space'])
onnx.checker.check_model(onnx_model)




class ModelInferer():
    def __init__(self, infer_config):
        self.logger = trt.Logger(trt.Logger.VERBOSE)
        self.builder = trt.Builder(logger)  # as an entrance of creating a neural network        
        
        
        pass

    def build_network(self):
        pass

    def build_engine(self):
        pass

