import os
from datetime import datetime as dt
from glob import glob

import cv2
import numpy as np
import tensorrt as trt
import torch as t
import torch.nn.functional as F
from cuda import cudart
from torch.autograd import Variable
from torchvision.models import resnet34


nHeight = 28
nWidth = 28
channels = 3
onnxFile = "./asset/model.onnx"
trtFile = "./asset/model.plan"


class Net(t.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = t.nn.Conv2d(3, 32, (5, 5), padding=(2, 2), bias=True)
        self.conv2 = t.nn.Conv2d(32, 64, (5, 5), padding=(2, 2), bias=True)
        self.fc1 = t.nn.Linear(64 * 7 * 7, 1024, bias=True)
        self.fc2 = t.nn.Linear(1024, 10, bias=True)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.reshape(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        z = t.argmax(x, dim=1)
        return z


class MyNet(t.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.model = resnet34()

    def forward(self, x):
        out = self.model(x)
        return x, out


# model = MyNet().cuda()
# model = Net().cuda()
model = resnet34().cuda()

t.onnx.export(model,
              t.randn(1, channels, nHeight, nWidth, device="cuda"),
              onnxFile,
              input_names=["x"], output_names=["z"],
              do_constant_folding=True,
              verbose=True,
              keep_initializers_as_inputs=False,
              opset_version=12,
              dynamic_axes={"x": {0: "nBatchSize"}, "z": {0: "nBatchSize"}})
print("Succeeded converting model into ONNX!")


# Parse network, rebuild network and do inference in TensorRT ------------------
logger = trt.Logger(trt.Logger.VERBOSE)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))


profile = builder.create_optimization_profile()
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)


parser = trt.OnnxParser(network, logger)
if not os.path.exists(onnxFile):
    print("Failed finding ONNX file!")
    exit()
print("Succeeded finding ONNX file!")
with open(onnxFile, "rb") as model:
    if not parser.parse(model.read()):
        print("Failed parsing .onnx file!")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()
    print("Succeeded parsing .onnx file!")

inputTensor = network.get_input(0)
# network.unmark_output(network.get_output(0))
profile.set_shape(inputTensor.name, [1, channels, nHeight, nWidth], [4, channels, nHeight, nWidth], [8, channels, nHeight, nWidth])
config.add_optimization_profile(profile)

engineString = builder.build_serialized_network(network, config)
if engineString is None:
    print("Failed building engine!")
    exit()
print("Succeeded building engine!")
with open(trtFile, "wb") as f:
    f.write(engineString)
engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

nIO = engine.num_io_tensors
lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

context = engine.create_execution_context()
context.set_input_shape(lTensorName[0], [1, 1, nHeight, nWidth])
for i in range(nIO):
    print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]), engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

bufferH = []
inferenceImage = "./src/test.jpg"
data = cv2.imread(inferenceImage)
data = cv2.resize(data, (nHeight, nWidth)).astype(np.float32).reshape(1, channels, nHeight, nWidth)
bufferH.append(np.ascontiguousarray(data))
for i in range(nInput, nIO):
    bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]), dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
bufferD = []
for i in range(nIO):
    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

for i in range(nInput):
    cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

for i in range(nIO):
    context.set_tensor_address(lTensorName[i], int(bufferD[i]))

context.execute_async_v3(0)

for i in range(nInput, nIO):
    cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

for i in range(nIO):
    print(lTensorName[i])
    print(bufferH[i])

for b in bufferD:
    cudart.cudaFree(b)

print("Succeeded running model in TensorRT!")
