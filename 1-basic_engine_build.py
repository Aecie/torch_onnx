"""
`1-basic_engine_build.py` demonstrates the fundamental workflow of using TensorRT's Python API to accelerate inference with these key steps:

1. **Engine Building**: Converts an ONNX model (`sample_model.onnx`) into a TensorRT engine, which is an optimized representation for faster inference.

2. **Engine Serialization**: Saves the built engine to disk (`sample_model.engine`), allowing it to be reused without rebuilding.

3. **Engine Loading**: Loads the serialized engine back from disk to verify it works correctly.

4. **Inference Execution**: Performs actual inference using TensorRT:
   - Creates an execution context from the engine
   - Prepares input/output tensors using the TensorRT 10.x API
   - Allocates GPU memory
   - Executes the model with random input data
   - Retrieves results back to the host

The file specifically uses TensorRT 10.x API conventions, including:
- Using `get_tensor_name()` and `get_tensor_shape()` instead of accessing bindings directly
- Using `set_tensor_address()` to specify input/output memory locations
- Using `execute_async_v3()` for executing inference

This demonstrates the basic end-to-end workflow from loading a model to performing optimized inference with TensorRT, serving as the foundation for more advanced capabilities shown in later examples.
"""
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from utils import build_engine, save_engine, load_engine  # Utility functions we'll create

def main():
    # Configuration
    onnx_path = "model_onnx/sample_model.onnx"
    engine_path = "model_onnx/sample_model.engine"
    
    # 1. Build TensorRT engine
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    
    # Build engine from ONNX
    engine = build_engine(builder, onnx_path)
    
    # 2. Serialize and save engine
    save_engine(engine, engine_path)
    
    # 3. Load engine back for verification
    loaded_engine = load_engine(logger, engine_path)
    
    # 4. Perform sample inference
    with loaded_engine.create_execution_context() as context:
        # Get input/output dimensions using TensorRT 10.x API
        input_name = loaded_engine.get_tensor_name(0)
        output_name = loaded_engine.get_tensor_name(1)
        
        # Get tensor shapes (returns tuple for static, Dims object for dynamic)
        input_shape = loaded_engine.get_tensor_shape(input_name)
        output_shape = loaded_engine.get_tensor_shape(output_name)
        
        # Convert Dims to tuple if dynamic
        if isinstance(input_shape, trt.Dims):
            input_shape = tuple(input_shape)
        if isinstance(output_shape, trt.Dims):
            output_shape = tuple(output_shape)
        
        print(f"Input shape: {input_shape}, Output shape: {output_shape}")
        
        # Create host and device buffers
        h_input = np.random.random(input_shape).astype(np.float32)
        d_input = cuda.mem_alloc(h_input.nbytes)
        h_output = np.empty(output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(h_output.nbytes)
        
        # Perform inference
        stream = cuda.Stream()
        
        # Set input/output addresses using TensorRT 10.x API
        context.set_tensor_address(input_name, int(d_input))
        context.set_tensor_address(output_name, int(d_output))
        
        # Execute without bindings parameter
        context.execute_async_v3(stream_handle=stream.handle)
        
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        
        print(f"Inference completed. Output shape: {h_output.shape}")

if __name__ == "__main__":
    main() 