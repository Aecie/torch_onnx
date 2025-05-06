"""
Common utilities for TensorRT operations
"""
import tensorrt as trt
import os
import pycuda.driver as cuda
import pycuda.autoinit  # Maintains context creation

def build_engine(builder, onnx_path):
    """Build TensorRT engine from ONNX file"""
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, builder.logger)
    
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file {onnx_path} not found")
    
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError("ONNX parsing failed")
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    # Get actual input dimensions from ONNX model
    input_tensor = network.get_input(0)
    input_shape = input_tensor.shape
    
    # Create optimization profile based on actual dimensions
    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_tensor.name,
        min=tuple([1] + list(input_shape[1:])),  # Keep original dimensions
        opt=tuple([1] + list(input_shape[1:])),   # Same as min/max for static
        max=tuple([1] + list(input_shape[1:]))    # All dimensions fixed
    )
    config.add_optimization_profile(profile)
    
    # Build and return engine
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine serialization failed")
    
    runtime = trt.Runtime(builder.logger)
    return runtime.deserialize_cuda_engine(serialized_engine)

def save_engine(engine, engine_path):
    """Serialize and save TensorRT engine"""
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

def load_engine(logger, engine_path):
    """Load serialized TensorRT engine"""
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def time_inference(context, iterations=100):
    """Time inference execution using CUDA events"""
    # Create CUDA events
    start_event = cuda.Event()
    end_event = cuda.Event()
    
    # Create stream for async timing
    stream = cuda.Stream()
    
    # Warm-up run
    context.execute_async_v2(bindings=[0]*context.engine.num_bindings, stream_handle=stream.handle)
    
    # Timing loop
    total_time = 0.0
    for _ in range(iterations):
        start_event.record(stream)
        context.execute_async_v2(bindings=[0]*context.engine.num_bindings, stream_handle=stream.handle)
        end_event.record(stream)
        end_event.synchronize()
        total_time += start_event.time_till(end_event)

    avg_time = total_time / iterations
    print(f"Average inference time over {iterations} iterations: {avg_time:.2f}ms")
    return avg_time
