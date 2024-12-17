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
import os
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver
from model_configs.train_infer import config

if not os.path.exists(config['onnx_engine_space']):
    os.mkdir(config['onnx_engine_space'])

class ModelInferer():
    def __init__(self, infer_config):
        self.model_name = infer_config['model_name']
        self.input_shape = infer_config['input_shape']
        self.output_shape = infer_config['output_shape']
        self.onnx_file_path = os.path.join(config['onnx_model_space'], '%s.onnx' % self.model_name)
        self.logger = trt.Logger(trt.Logger.WARNING)
    
    def proceed(self):
        """
        Serialization
            build the engine
            serialize the engine

        Deserialization
            read serialized engine
            create the runtime
            execute inference

        """
        pass

    def serialization(self):
        # declare trt builder, network and parser
        # builder: to build an engine, but populated network should be provided
        # network: a computational graph
        # parser: populate a network with trained onnx model
        # engine: output of trt optimizer, generates context to perform inference, usually in .trt, .engine and .plan formula, the three are equivalent
        # engine config: specify the way of optimization
        with trt.Builder(logger=self.logger) as builder, \
             builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
             trt.OnnxParser(network=network, logger=self.logger) as parser, \
             builder.create_builder_config() as engine_config:
            # populate the network
            assert parser.parse_from_file(self.onnx_file_path)
            # set optimization level
            engine_config.builder_optimization_level = 5  # max level   
            # create trt engine with config
            engine = builder.build_engine_with_config(network=network, config=engine_config)
            with open(os.path.join(config['onnx_engine_space'], '%s.engine' % self.model_name), 'wb') as engine_file_handler:
                engine_file_handler.write(engine.serialize())  # write the engine data (engine serialized output)
            print('saved engine')

    def deserialization(self, input_data):
        output_data = np.empty(self.output_shape, dtype=np.float32)
        # runtime: responsible for trt engine executiion
        # context: manage the trt inference execution
        runtime = trt.Runtime(self.logger)
        with open(os.path.join(config['onnx_engine_space'], '%s.engine' % self.model_name), 'rb') as engine_file_handler:
            engine_data = engine_file_handler.read()
            engine = runtime.deserialize_cuda_engine(engine_data)
            context = engine.create_execution_context()

            input_device_allocate, output_device_allocate = self.cuda_memory_allocation()
            pycuda.driver.memcpy_htod(input_device_allocate, input_data)
            context.execute_v2([int(input_device_allocate), int(output_device_allocate)])
            pycuda.driver.memcpy_dtoh(output_data, output_device_allocate)
        return output_data

    def cuda_memory_allocation(self):
        input_size = trt.volume(self.input_shape) * trt.float32.itemsize
        output_size = trt.volume(self.output_shape) * trt.float32.itemsize
        input_device_allocate = pycuda.driver.mem_alloc(input_size)
        output_device_allocate = pycuda.driver.mem_alloc(output_size)
        return input_device_allocate, output_device_allocate



inferer = ModelInferer(infer_config=config['AlexNet_MNIST'])
inferer.serialization()
