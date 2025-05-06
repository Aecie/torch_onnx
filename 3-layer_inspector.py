"""
Demonstrates TensorRT network layer inspection
Features covered:
- Network layer iteration and visualization
- Layer type identification and properties
- Parameter extraction and analysis
- Layer fusion observation and optimization hints
- Precision analysis
"""
import tensorrt as trt
import numpy as np
import os

def inspect_network_topology(network):
    """Visualize network topology with connections"""
    print("\n===== NETWORK TOPOLOGY =====")
    print(f"Network name: {network.name}")
    print(f"Total layers: {network.num_layers}")
    print(f"Total inputs: {network.num_inputs}")
    print(f"Total outputs: {network.num_outputs}")
    
    # Map tensor names to producer layers
    tensor_producers = {}
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        for j in range(layer.num_outputs):
            tensor = layer.get_output(j)
            tensor_producers[tensor.name] = (i, layer.name)
    
    # Show layer connections
    print("\n----- Layer Connections -----")
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        print(f"\nLayer {i}: {layer.name} ({layer.type})")
        
        # Show inputs and their sources
        for j in range(layer.num_inputs):
            tensor = layer.get_input(j)
            if not tensor:
                continue
                
            producer = tensor_producers.get(tensor.name, ("NETWORK_INPUT", "INPUT"))
            print(f"  ↳ Input {j}: {tensor.name} from {producer[1]}")
            
        # Show outputs and shapes
        for j in range(layer.num_outputs):
            tensor = layer.get_output(j)
            if tensor:
                print(f"  ↳ Output {j}: {tensor.name} | Shape: {tensor.shape}")

def analyze_layer_parameters(network):
    """Analyze specific parameters for different layer types"""
    print("\n===== LAYER PARAMETERS =====")
    
    # Track layer type statistics
    layer_types = {}
    
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        
        # Count layer types
        layer_type = str(layer.type)
        layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
        
        print(f"\nLayer {i}: {layer.name} ({layer.type})")
        
        # Show input/output tensor details
        for j in range(layer.num_inputs):
            tensor = layer.get_input(j)
            if tensor:
                print(f"  ↳ Input {j}: {tensor.name} | Shape: {tensor.shape} | Type: {tensor.dtype}")
        
        for j in range(layer.num_outputs):
            tensor = layer.get_output(j)
            if tensor:
                print(f"  ↳ Output {j}: {tensor.name} | Shape: {tensor.shape} | Type: {tensor.dtype}")
        
        # Show layer-specific attributes through safe string representation
        print(f"  ↳ Precision: {layer.precision}")
    
    # Summarize layer types
    print("\n----- Layer Type Summary -----")
    for layer_type, count in sorted(layer_types.items(), key=lambda x: x[1], reverse=True):
        print(f"{layer_type}: {count} layers")

def check_fusion_opportunities(network):
    """Check for potential layer fusion opportunities"""
    print("\n===== FUSION OPPORTUNITIES =====")
    
    # Common fusion patterns
    fusion_patterns = [
        ("Conv+ReLU", 
         [trt.LayerType.CONVOLUTION, trt.LayerType.ACTIVATION]),
        ("Conv+BatchNorm", 
         [trt.LayerType.CONVOLUTION, trt.LayerType.SCALE]),
        ("Conv+ElementWise", 
         [trt.LayerType.CONVOLUTION, trt.LayerType.ELEMENTWISE]),
        ("FC+ReLU", 
         [trt.LayerType.MATRIX_MULTIPLY, trt.LayerType.ACTIVATION])
    ]
    
    for i in range(network.num_layers - 1):
        layer = network.get_layer(i)
        next_layer = network.get_layer(i + 1)
        
        # Check if this layer's output connects to next layer's input
        connected = False
        for j in range(layer.num_outputs):
            out_tensor = layer.get_output(j)
            for k in range(next_layer.num_inputs):
                if next_layer.get_input(k) == out_tensor:
                    connected = True
                    break
            if connected:
                break
                
        if connected:
            # Check against fusion patterns
            for name, pattern in fusion_patterns:
                if layer.type == pattern[0] and next_layer.type == pattern[1]:
                    print(f"Potential {name} fusion: {layer.name} → {next_layer.name}")

def main():
    print("TensorRT Layer Inspector Demonstration")
    
    onnx_path = "model_onnx/sample_model.onnx"
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model not found at {onnx_path}")
        return
        
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    
    # Configure network for inspection
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("Error parsing ONNX model:")
            for error in range(parser.num_errors):
                print(f"  - {parser.get_error(error)}")
            return
    
    # Perform comprehensive network analysis
    inspect_network_topology(network)
    analyze_layer_parameters(network)
    check_fusion_opportunities(network)
    
    print("\nLayer inspection complete!")

if __name__ == "__main__":
    main() 