import time
import cv2 as cv
import numpy as np
from cuda import cudart
import tensorrt as trt

from serialize_onnx import config


def load_plan(args):
    # logger记录运行过程中的信息
    logger = trt.Logger(trt.Logger.VERBOSE)

    # 读取序列化文件
    with open(args.trtFile, "rb") as f:
        engineString = f.read()

    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    nIO = engine.num_io_tensors
    lTensorName = [engine.get_tensor_name(i) for i in range(nIO)]
    nInput = [engine.get_tensor_mode(lTensorName[i]) for i in range(nIO)].count(trt.TensorIOMode.INPUT)

    context = engine.create_execution_context()
    context.set_input_shape(lTensorName[0], [1, args.C, args.img_h, args.img_w])
    for i in range(nIO):
        print("[%2d]%s->" % (i, "Input " if i < nInput else "Output"), engine.get_tensor_dtype(lTensorName[i]),
              engine.get_tensor_shape(lTensorName[i]), context.get_tensor_shape(lTensorName[i]), lTensorName[i])

    bufferH = []
    inferenceImage = "./src/test.jpg"
    data = cv.imread(inferenceImage)
    data = cv.cvtColor(data, cv.COLOR_BGR2RGB)
    data = cv.resize(data, (args.img_h, args.img_w)).astype(np.float32).reshape(1, args.C, args.img_h, args.img_w)

    bufferH.append(np.ascontiguousarray(data))
    for i in range(nInput, nIO):
        bufferH.append(np.empty(context.get_tensor_shape(lTensorName[i]),
                                dtype=trt.nptype(engine.get_tensor_dtype(lTensorName[i]))))
    bufferD = []
    for i in range(nIO):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    for i in range(nIO):
        context.set_tensor_address(lTensorName[i], int(bufferD[i]))

    start = time.time()
    context.execute_async_v3(0)
    duration = time.time() - start
    for i in range(nInput, nIO):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    for i in range(nInput, nIO):
        print(lTensorName[i])
        print(bufferH[i].argmax(-1))

    for b in bufferD:
        cudart.cudaFree(b)
    print(f"inference time is {duration} s")
    print("Succeeded running model in TensorRT!")


if __name__ == '__main__':
    args = config()
    load_plan(args)