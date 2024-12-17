import onnx_graphsurgeon as gs
import numpy as np
import tensorrt as trt
import os



X = gs.Variable(name="X", dtype=np.float32, shape=('n_rows', 2))
Y = gs.Variable(name="Y", dtype=np.float32, shape=('n_rows', 2))
Z = gs.Variable(name="Z", dtype=np.float32, )
node = gs.Node(op="Concat", inputs=[X, Y], outputs=[Z], attrs={"axis": 0})
graph = gs.Graph(nodes=[node], inputs=[X, Y], outputs=[Z])
model = gs.export_onnx(graph)

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(0)
parser = trt.OnnxParser(network, logger)

print(type(model.SerializeToString()))
assert(parser.parse(model.SerializeToString()))

config = builder.create_builder_config()
profile = builder.create_optimization_profile()
x_dim0 = {1}
y_dim0 = {1}
profile.set_shape("X", (1, 2), (1, 2), (1, 2))
profile.set_shape("Y", (1, 2), (1, 2), (1, 2))
config.add_optimization_profile(profile)
serialized_engine = builder.build_serialized_network(network, config)
# with open("tmp3.engine", 'wb') as f:
#     f.write(serialized_engine)

with open(os.path.join('../onnx_engines', 'tmp.engine'), 'wb') as f:
    f.write(serialized_engine)