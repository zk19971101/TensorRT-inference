#include<cuda.h>
#include<cuda_runtime_api.h>
#include<stdlib.h>
#include<opencv.hpp>
#include<device_launch_parameters.h>

#include<NvInfer.h>
#include<iostream>
#include<fstream>
#include<string>


using namespace nvinfer1;
using namespace std;

size_t dataTypeToSize(DataType dataType);


class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}gLogger;

__global__ void process(cv::cuda::PtrStepSz<uchar3>src, float *dst)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int width = src.cols;
    const int height = src.rows;
    const int size = width * height;
    float mean[3] = {};
    float std[3] = {1.0, 1.0, 1.0};
    if (x < src.cols && y < src.rows)
    {
        int index = x + y * width;
        dst[index] = (float(src(y, x).z) - mean[0]) / std[0];
        dst[index+size] = (float(src(y, x).y) - mean[1]) / std[1];
        dst[index+2*size] = (float(src(y, x).x) - mean[2]) / std[2];
    }

}

int main()
{
    //读取保存的二进制模型
    const std::string trtFile{ "./model.plan" };
    std::ifstream engineFile(trtFile, std::ios::binary);
    long int      fsize = 0;
    engineFile.seekg(0, engineFile.end);
    fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);
    std::vector<char> engineString(fsize);
    engineFile.read(engineString.data(), fsize);

    // 判断是否读取成果
    if (engineString.size() == 0)
    {
        std::cout << "Failed getting serialized engine!" << std::endl;
        return -1;
    }
    std::cout << "Succeeded getting serialized engine!" << std::endl;

    // 基于读取的文件构建推理引擎
    ICudaEngine* engine = nullptr;
    IRuntime* runtime{ createInferRuntime(gLogger) };
    engine = runtime->deserializeCudaEngine(engineString.data(), fsize);

    // 判断推理引擎是否构建成果
    if (engine == nullptr)
    {
        std::cout << "Failed loading engine!" << std::endl;
        return -2;
    }
    std::cout << "Succeeded loading engine!" << std::endl;

    // 获取模型的输入、输出数量和对应名字
    long unsigned int        nIO = engine->getNbIOTensors();
    long unsigned int        nInput = 0;
    long unsigned int        nOutput = 0;
    std::vector<std::string> vTensorName(nIO);
    for (int i = 0; i < nIO; ++i)
    {
        vTensorName[i] = std::string(engine->getIOTensorName(i));
        nInput += int(engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kINPUT);
        nOutput += int(engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kOUTPUT);
    }

    // 构建执行推理过程的context
    IExecutionContext* context = engine->createExecutionContext();

    // 设置执行推理的输入数据形状、占用显存大小
    context->setInputShape(vTensorName[0].c_str(), Dims32{ 4, {1, 3, 256, 256} });


    std::vector<int> vTensorSize(nIO, 0);
    for (int i = 0; i < nIO; ++i)
    {
        Dims32 dim = context->getTensorShape(vTensorName[i].c_str());
        int    size = 1;
        for (int j = 0; j < dim.nbDims; ++j)
        {
            size *= dim.d[j];
        }
        vTensorSize[i] = size * dataTypeToSize(engine->getTensorDataType(vTensorName[i].c_str()));
    }

    // https://www.dotndash.net/2023/03/09/using-tensorrt-with-opencv-cuda.html
    float* src, *res;
    cudaMalloc((void**)&src, 256 * 256 * 3 * sizeof(float));
    cudaMalloc((void**)&res, 1000 * sizeof(float));
    context->setTensorAddress("image", src);
    context->setTensorAddress("label", res);

    cv::Mat img = cv::imread("test.jpg");
    cv::resize(img, img, cv::Size(256, 256));
    //img.convertTo(img, CV_32FC3, 1.0 / 255.);
    cv::cuda::GpuMat img_d(img);
    dim3 block_size(32, 32);
    dim3 grid_size((img.cols + 32 - 1) / 32, (img.rows + 32 - 1) / 32);
   
    process << <grid_size, block_size >> > (img_d, res);


    context->enqueueV3(0);

    float* res_h = new float[1000];
    cudaMemcpy(res_h, res, 1000 * sizeof(float), cudaMemcpyDeviceToHost);

    float max_prob = 0.0;
    int max_index = 0;
    for (int i = 0; i < 1000; i++)
    {
        if (res_h[i] > max_prob)
        {
            max_prob = res_h[i];
            max_index = i;
        }
    }
    cout << "index:" << max_index << "\tpro:" <<max_prob << endl;
    delete[]res_h;
    cudaFree(src);
    cudaFree(res);


	return 0;
}


// get the size in byte of a TensorRT data type
size_t dataTypeToSize(DataType dataType)
{
    switch (dataType)
    {
    case DataType::kFLOAT:
        return 4;
    case DataType::kHALF:
        return 2;
    case DataType::kINT8:
        return 1;
    case DataType::kINT32:
        return 4;
    case DataType::kBOOL:
        return 1;
    case DataType::kUINT8:
        return 1;
    case DataType::kFP8:
        return 1;
    default:
        return 4;
    }
}