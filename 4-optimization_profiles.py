"""
Demonstrates optimization profile usage
Features covered:
- Multiple optimization profiles
- Profile selection at runtime
- Dynamic shape performance optimization
"""
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import os
from utils import save_engine

def main():
    onnx_path = "model_onnx/dynamic_model.onnx"
    
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model not found at {onnx_path}")
        return
        
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    
    # Build engine with multiple profiles
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("Error parsing ONNX model:")
            for error in range(parser.num_errors):
                print(f"  - {parser.get_error(error)}")
            return
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    # Create three optimization profiles
    input_tensor = network.get_input(0)
    print(f"Input tensor name: {input_tensor.name}, shape: {input_tensor.shape}")
    
    # Store profile configs for later reference
    profiles = []
    for profile_idx in range(3):
        profile = builder.create_optimization_profile()
        min_shape = (2**profile_idx, 3, 224, 224)
        opt_shape = (2**(profile_idx+1), 3, 224, 224)
        max_shape = (2**(profile_idx+2), 3, 224, 224)
        
        profiles.append((min_shape, opt_shape, max_shape))
        print(f"Profile {profile_idx}: min={min_shape}, opt={opt_shape}, max={max_shape}")
        profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
    
    # Build engine
    serialized_engine = builder.build_serialized_network(network, config)
    engine = trt.Runtime(logger).deserialize_cuda_engine(serialized_engine)
    
    if not engine:
        print("Failed to build engine")
        return
        
    print(f"Engine built successfully with {engine.num_optimization_profiles} profiles")
    
    # Save the engine for future use
    engine_path = "model_onnx/dynamic_model_multiprofile.engine"
    save_engine(engine, engine_path)
    print(f"Engine saved to {engine_path}")
    
    # Demonstrate profile selection - we need to create separate contexts for each profile
    for profile_idx in range(engine.num_optimization_profiles):
        print(f"\n--- Testing Profile {profile_idx} ---")
        # Create context with specific profile
        with engine.create_execution_context() as context:
            # In TensorRT 10, we need to set the profile index during context creation
            # or use a different method to select the profile
            try:
                # Try different approaches depending on TensorRT version
                if hasattr(context, "set_optimization_profile_async"):
                    context.set_optimization_profile_async(profile_idx, cuda.Stream().handle)
                    print(f"Using set_optimization_profile_async for profile {profile_idx}")
                else:
                    # For newer TensorRT that doesn't have active_optimization_profile attribute
                    # Create a new execution context specifically for this profile
                    print(f"Using profile {profile_idx} for this execution context")
                
                # Get valid batch sizes for this profile
                min_batch, opt_batch, max_batch = profiles[profile_idx][0][0], profiles[profile_idx][1][0], profiles[profile_idx][2][0]
                print(f"Valid batch sizes for profile {profile_idx}: {min_batch} to {max_batch}")
                
                # Test with a valid batch size for this profile
                test_batch = min(max(min_batch, 2**profile_idx), max_batch)
                input_name = engine.get_tensor_name(0)
                output_name = engine.get_tensor_name(1)
                shape = (test_batch, 3, 224, 224)
                
                # Set shape for inference
                context.set_input_shape(input_name, shape)
                
                # Allocate memory
                h_input = np.random.random(shape).astype(np.float32)
                d_input = cuda.mem_alloc(h_input.nbytes)
                
                # Get output shape and allocate output memory
                output_shape = context.get_tensor_shape(output_name)
                h_output = np.empty(output_shape, dtype=np.float32)
                d_output = cuda.mem_alloc(h_output.nbytes)
                
                # Perform inference
                stream = cuda.Stream()
                cuda.memcpy_htod_async(d_input, h_input, stream)
                
                # Set tensor addresses
                context.set_tensor_address(input_name, int(d_input))
                context.set_tensor_address(output_name, int(d_output))
                
                # Execute inference
                context.execute_async_v3(stream.handle)
                
                # Copy results back
                cuda.memcpy_dtoh_async(h_output, d_output, stream)
                stream.synchronize()
                
                print(f"Inference completed with batch size {test_batch} using profile {profile_idx}")
                print(f"Output shape: {output_shape}")
                
            except Exception as e:
                print(f"Error using profile {profile_idx}: {e}")

if __name__ == "__main__":
    main() 