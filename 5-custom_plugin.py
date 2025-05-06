"""
Demonstrates custom plugin implementation
Features covered:
- Plugin creation
- Plugin registration
- Custom layer integration
"""
import tensorrt as trt
import numpy as np

# Custom plugin implementation
class LeakyReLUPlugin(trt.IPluginV2DynamicExt):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = np.float32(alpha)
        self.name = "LeakyReLUPlugin"
        
    def initialize(self):
        return 0
    
    def terminate(self):
        pass
    
    def get_output_datatype(self, index, input_types):
        return input_types[0]
    
    def configure_plugin(self, in_out, in_pos, out_pos):
        return 0
    
    def serialize(self):
        return bytearray(self.alpha.tobytes())
    
    def attach_to_context(self, ctxt):
        pass
    
    def clone(self):
        return LeakyReLUPlugin(self.alpha)
    
    def get_plugin_namespace(self):
        return ""
    
    # Other required methods omitted for brevity...

def main():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # Create input
    input_tensor = network.add_input(name="input", dtype=trt.float32, shape=(1, 3, 224, 224))
    
    # Add custom plugin layer
    plugin_creator = trt.get_plugin_registry().get_plugin_creator("LeakyReLUPlugin", "1")
    if not plugin_creator:
        print("Plugin not found, registering...")
        # In real usage, need proper plugin registration
        trt.get_plugin_registry().register_creator(LeakyReLUPluginCreator(), "")
    
    # Build network with custom plugin
    plugin_layer = network.add_plugin_v2([input_tensor], plugin_creator.create_plugin("LeakyReLUPlugin", trt.PluginFieldCollection()))
    
    # Mark output
    plugin_layer.get_output(0).name = "output"
    network.mark_output(plugin_layer.get_output(0))
    
    # Build engine
    config = builder.create_builder_config()
    engine = builder.build_engine(network, config)
    
    print("Engine with custom plugin built successfully!")

if __name__ == "__main__":
    main() 