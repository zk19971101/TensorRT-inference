#include<cuda.h>
#include<cuda_runtime_api.h>
#include<stdlib.h>
#include<NvInfer.h>
#include<iostream>
#include<fstream>
#include<string>
#include<opencv.hpp>


using namespace nvinfer1;
using namespace std;

std::string dataTypeToString(DataType dataType);
size_t dataTypeToSize(DataType dataType);
std::string shapeToString(Dims32 dim);

class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}gLogger;


int main()
{
    ICudaEngine* engine = nullptr;
    const std::string trtFile{ "../asset/model.plan" };

    std::ifstream engineFile(trtFile, std::ios::binary);
    long int      fsize = 0;

    engineFile.seekg(0, engineFile.end);
    fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);
    std::vector<char> engineString(fsize);
    engineFile.read(engineString.data(), fsize);

    if (engineString.size() == 0)
    {
        std::cout << "Failed getting serialized engine!" << std::endl;
        return -1;
    }
    std::cout << "Succeeded getting serialized engine!" << std::endl;
    vector<int> data_int;

    IRuntime* runtime{ createInferRuntime(gLogger) };
    engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
    if (engine == nullptr)
    {
        std::cout << "Failed loading engine!" << std::endl;
        return -2;
    }
    std::cout << "Succeeded loading engine!" << std::endl;


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

    IExecutionContext* context = engine->createExecutionContext();
    context->setInputShape(vTensorName[0].c_str(), Dims32{ 4, {1, 3, 256, 256} });

    for (int i = 0; i < nIO; ++i)
    {
        std::cout << std::string(i < nInput ? "Input [" : "Output[");
        std::cout << i << std::string("]-> ");
        std::cout << dataTypeToString(engine->getTensorDataType(vTensorName[i].c_str())) << std::string(" ");
        std::cout << shapeToString(engine->getTensorShape(vTensorName[i].c_str())) << std::string(" ");
        std::cout << shapeToString(context->getTensorShape(vTensorName[i].c_str())) << std::string(" ");
        std::cout << vTensorName[i] << std::endl;
    }

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

    vector<void*>vBufferH(nIO, nullptr);
    vector<void*>vBufferD(nIO, nullptr);
    for (int i = 0; i < nIO; ++i)
    {
        vBufferH[i] = (void*)new char[vTensorSize[i]];
        cudaMalloc(&vBufferD[i], vTensorSize[i]);
    }

    //float* pData = (float*)vBufferH[0];

    //for (int i = 0; i < vTensorSize[0] / dataTypeToSize(engine->getTensorDataType(vTensorName[0].c_str())); ++i)
    //{
    //    pData[i] = float(i);
    //}
    for (int i = 0; i < nInput; ++i)
    {
        cudaMemcpy(vBufferD[i], vBufferH[i], vTensorSize[i], cudaMemcpyHostToDevice);
    }

    for (int i = 0; i < nIO; ++i)
    {
        context->setTensorAddress(vTensorName[i].c_str(), vBufferD[i]);
    }

    context->enqueueV3(0);

    for (int i = nInput; i < nIO; ++i)
    {
        cudaMemcpy(vBufferH[i], vBufferD[i], vTensorSize[i], cudaMemcpyDeviceToHost);
    }

    //for (int i = 0; i < nIO; ++i)
    //{
    //    printArrayInformation((float*)vBufferH[i], context->getTensorShape(vTensorName[i].c_str()), vTensorName[i], true, true);
    //}

    for (int i = 0; i < nIO; ++i)
    {
        delete[](char*)vBufferH[i];
        cudaFree(vBufferD[i]);
    }


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

// get the string of a TensorRT shape
 std::string shapeToString(Dims32 dim)
{
    std::string output("(");
    if (dim.nbDims == 0)
    {
        return output + std::string(")");
    }
    for (int i = 0; i < dim.nbDims - 1; ++i)
    {
        output += std::to_string(dim.d[i]) + std::string(", ");
    }
    output += std::to_string(dim.d[dim.nbDims - 1]) + std::string(")");
    return output;
}

// get the string of a TensorRT data type
std::string dataTypeToString(DataType dataType)
{
    switch (dataType)
    {
    case DataType::kFLOAT:
        return std::string("FP32 ");
    case DataType::kHALF:
        return std::string("FP16 ");
    case DataType::kINT8:
        return std::string("INT8 ");
    case DataType::kINT32:
        return std::string("INT32");
    case DataType::kBOOL:
        return std::string("BOOL ");
    default:
        return std::string("Unknown");
    }
}