"""
`2-dynamic_shapes.py` demonstrates how to handle dynamic input dimensions in TensorRT, which is crucial for models that need to process inputs of varying sizes. This file:

1. **Creates Multiple Dynamic Engines**: Builds TensorRT engines with different optimization profiles that handle various types of dynamic dimensions:
   - `batch_dynamic`: Supports varying batch sizes while keeping spatial dimensions constant
   - `spatial_dynamic`: Supports varying image sizes (height/width) with fixed batch size
   - `fully_dynamic`: Supports varying both batch size and spatial dimensions

2. **Sets Optimization Profiles**: Each profile defines valid ranges (min, optimal, max) for tensor dimensions to optimize performance across different input shapes.

3. **Demonstrates Runtime Shape Handling**: Shows how to:
   - Set specific input shapes at runtime
   - Query the resulting output shapes dynamically
   - Allocate appropriate memory for those shapes
   - Process inputs of different dimensions with the same engine

4. **Tests Multiple Dimensions**: Runs inference with various input shapes to demonstrate how TensorRT adapts:
   - For `batch_dynamic`: Tests batch sizes 1, 4, and 8
   - For `spatial_dynamic`: Tests resolutions 224x224, 512x512, and 1024x1024
   - For `fully_dynamic`: Tests combinations of both

5. **Uses TensorRT 10.x API**: Employs current API patterns like `set_input_shape()`, `get_tensor_shape()`, and `execute_async_v3()`.

This file is particularly valuable for applications like computer vision, NLP, or any scenario where input sizes vary (e.g., different image resolutions, varying sequence lengths, or fluctuating batch sizes).

"""
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from utils import build_engine, save_engine, load_engine

def build_dynamic_engine(builder, onnx_path, profile_name):
    """Build engine with specific dynamic shape profile"""
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, builder.logger)
    
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError("ONNX parsing failed")
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    
    input_tensor = network.get_input(0)
    
    # Handle dynamic dimensions from ONNX (-1)
    input_dims = []
    for dim in input_tensor.shape[1:]:  # Skip batch dimension
        if dim == -1:
            input_dims.append(224)  # Replace dynamic dim with default
        else:
            input_dims.append(dim)
    
    # Define profiles with concrete dimensions
    profiles = {
        "batch_dynamic": {
            "min": (1, *input_dims),
            "opt": (4, *input_dims),
            "max": (8, *input_dims)
        },
        "spatial_dynamic": {
            "min": (1, input_dims[0], 224, 224),
            "opt": (1, input_dims[0], 512, 512),
            "max": (1, input_dims[0], 1024, 1024)
        },
        "fully_dynamic": {
            "min": (1, input_dims[0], 128, 128),
            "opt": (2, input_dims[0], 256, 256),
            "max": (4, input_dims[0], 512, 512)
        }
    }
    
    # Validate profile selection
    if profile_name not in profiles:
        raise ValueError(f"Invalid profile: {profile_name}. Choose from {list(profiles.keys())}")
    
    # Create and configure profile
    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_tensor.name,
        min=profiles[profile_name]["min"],
        opt=profiles[profile_name]["opt"],
        max=profiles[profile_name]["max"]
    )
    config.add_optimization_profile(profile)
    
    serialized_engine = builder.build_serialized_network(network, config)
    return trt.Runtime(builder.logger).deserialize_cuda_engine(serialized_engine)

def test_dynamic_inference(engine, test_shapes):
    """Test engine with various input shapes"""
    with engine.create_execution_context() as context:
        input_name = engine.get_tensor_name(0)
        output_name = engine.get_tensor_name(1)
        
        for shape in test_shapes:
            # Set dynamic shape
            context.set_input_shape(input_name, shape)
            
            # Get output shape after setting input shape
            output_shape = context.get_tensor_shape(output_name)
            
            # Allocate buffers
            h_input = np.random.random(shape).astype(np.float32)
            d_input = cuda.mem_alloc(h_input.nbytes)
            h_output = np.empty(output_shape, dtype=np.float32)
            d_output = cuda.mem_alloc(h_output.nbytes)
            
            # Inference
            stream = cuda.Stream()
            cuda.memcpy_htod_async(d_input, h_input, stream)
            context.set_tensor_address(input_name, int(d_input))
            context.set_tensor_address(output_name, int(d_output))
            context.execute_async_v3(stream.handle)
            
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            stream.synchronize()
            
            print(f"Shape {shape} completed. Output shape: {output_shape}")

def main():
    onnx_path = "model_onnx/dynamic_model.onnx"
    
    # Build and save different dynamic engines
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    
    # Create multiple engines for different profiles
    for profile_name in ["batch_dynamic", "spatial_dynamic", "fully_dynamic"]:
        engine = build_dynamic_engine(builder, onnx_path, profile_name)
        save_engine(engine, f"model_onnx/dynamic_model_{profile_name}.engine")
        
        # Test dynamic shapes
        test_shapes = {
            "batch_dynamic": [(1,3,224,224), (4,3,224,224), (8,3,224,224)],
            "spatial_dynamic": [(1,3,224,224), (1,3,512,512), (1,3,1024,1024)],
            "fully_dynamic": [(1,3,128,128), (2,3,256,256), (4,3,512,512)]
        }
        
        print(f"\nTesting {profile_name} engine:")
        test_dynamic_inference(engine, test_shapes[profile_name])

if __name__ == "__main__":
    main() 