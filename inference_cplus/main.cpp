//#include<cuda.h>
//#include<cuda_runtime_api.h>
//#include<stdlib.h>
//#include<NvInfer.h>
//#include<iostream>
//#include<NvOnnxParser.h>
//
//using namespace nvonnxparser;
//using namespace nvinfer1;
//using namespace std;
//
//class Logger : public ILogger
//{
//    void log(Severity severity, const char* msg) noexcept override
//    {
//        // suppress info-level messages
//        if (severity <= Severity::kWARNING)
//            std::cout << msg << std::endl;
//    }
//} ;
//
//int main()
//{
//    // ����logger
//    Logger logger;
//
//    //����Builder
//    IBuilder* builder = createInferBuilder(logger);
//
//    //��build����network
//    uint32_t flag = 1U << static_cast<uint32_t>
//        (NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
//    INetworkDefinition* network = builder->createNetworkV2(flag);
//
//    //��build����config
//    IBuilderConfig* config = builder->createBuilderConfig();
//    config->setFlag(BuilderFlag::kFP16);
//    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 20);
//
//    //��build����profile
//    IOptimizationProfile* profile = builder->createOptimizationProfile();
//
//
//    // ͨ��parser��.onnxģ�ͽ��ж�ȡ
//    char modelFile[] = "../asset/resnet.onnx";
//    IParser* parser = createParser(*network, logger);
//    parser->parseFromFile(modelFile,
//        static_cast<int32_t>(ILogger::Severity::kWARNING));
//    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
//    {
//        std::cout << parser->getError(i)->desc() << std::endl;
//    }
//
//    //ͨ��profile����dynamic shape
//    ITensor* inputTensor = network->getInput(0);
//    const int         nHeight{ 256 };
//    const int         nWidth{ 256 };
//    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims32{ 4, {1, 3, nHeight, nWidth} });
//    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims32{ 4, {4, 3, nHeight, nWidth} });
//    profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims32{ 4, {8, 3, nHeight, nWidth} });
//    config->addOptimizationProfile(profile);
//
//    //network->unmarkOutput(*network->getOutput(0));
//
//    //ģ�ͽ������л�ת��ΪTRT�м��ʽ��ͨ��build��network��configת���������ļ�Ϊ.plan��ʽ
//    IHostMemory* engineString = builder->buildSerializedNetwork(*network, *config);
//    if (engineString == nullptr || engineString->size() == 0)
//    {
//        std::cout << "Failed building serialized engine!" << std::endl;
//        return -1;
//    }
//    std::cout << "Succeeded building serialized engine!" << std::endl;
//    
//   
//    std::ofstream engineFile(trtFile, std::ios::binary);
//    if (!engineFile)
//    {
//        std::cout << "Failed opening file to write" << std::endl;
//        return;
//    }
//    engineFile.write(static_cast<char*>(engineString->data()), engineString->size());
//    if (engineFile.fail())
//    {
//        std::cout << "Failed saving .plan file!" << std::endl;
//        return;
//    }
//    std::cout << "Succeeded saving .plan file!" << std::endl;
//}
//
//    
//
//    delete parser;
//    delete network;
//    delete config;
//    delete builder;
//    delete engineString;
//
//
//	return 0;
//}