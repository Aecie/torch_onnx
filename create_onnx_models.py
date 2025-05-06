"""
Script to generate sample ONNX models for TensorRT tutorials
"""
import torch
import torch.nn as nn
import onnx
import torch.nn.functional as F

def create_sample_model():
    """Create a simple CNN model for basic tutorials"""
    class SimpleCNN(nn.Module):
        """
        X (3, 224, 224) -> 
        Conv2d (16, 3, 3, padding=1) -> ReLU -> 
        MaxPool2d (2, 2) -> 
        Linear (10) -> 
        Output (1, 10)
        """
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(16 * 112 * 112, 10)
            
        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = x.view(-1, 16 * 112 * 112)
            x = self.fc1(x)
            return x
            
    model = SimpleCNN()
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        "model_onnx/sample_model.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}}  # Dynamic batch size
    )
    print("Sample model exported to model_onnx/sample_model.onnx")

def create_dynamic_shape_model():
    """Create model with dynamic input dimensions"""
    class DynamicModel(nn.Module):
        """
        X (3, 256, 256) -> 
        Conv2d (16, 3, 3, padding=1) -> ReLU -> 
        AvgPool2d (1, 2) -> 
        Output (1, 16)
        """
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            
            # Fixed window size based on expected input ratio
            # Assumes input width is multiple of 128
            x = F.avg_pool2d(x, kernel_size=(1, x.size(3)//128), stride=(1, x.size(3)//128))
            return x

    model = DynamicModel()
    dummy_input = torch.randn(1, 3, 256, 256)  # Ensure input is multiple of 128
    
    # Register custom symbolic function for dynamic pooling
    def avg_pool2d_symbolic(g, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override):
        # Handle dynamic kernel/stride by using fixed ratio
        kernel = [1, 2]  # Matches 256//128=2 ratio in dummy input
        strides = [1, 2]
        return g.op("AveragePool", input, 
                   kernel_shape_i=kernel,
                   strides_i=strides,
                   ceil_mode_i=0)
    
    torch.onnx.register_custom_op_symbolic('::avg_pool2d', avg_pool2d_symbolic, 11)

    torch.onnx.export(
        model,
        dummy_input,
        "model_onnx/dynamic_model.onnx",
        export_params=True,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {
                0: "batch_size",
                2: "height",
                3: "width"
            }
        }
    )
    print("Dynamic model exported to model_onnx/dynamic_model.onnx")

if __name__ == "__main__":
    # Create model directory if not exists
    from pathlib import Path
    Path("model_onnx").mkdir(exist_ok=True)
    
    # Generate sample models
    create_sample_model()
    create_dynamic_shape_model()
    
    # Verify ONNX models
    for model_path in ["model_onnx/sample_model.onnx", "model_onnx/dynamic_model.onnx"]:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print(f"{model_path} is valid ONNX (IR version: {model.ir_version})") 