
/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <dlfcn.h>
#include "NvInfer.h"
#include <time.h>
#include <cuda_fp16.h>
#include "TrtLite.h"
#include "Utils.h"
#include <cuda_runtime.h>
using namespace nvinfer1;

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger(simplelogger::TRACE);


static void ConfigBuilderProc(IBuilderConfig *config, vector<IOptimizationProfile *> vProfile, void *pData) {
    BuildEngineParam *pParam = (BuildEngineParam *)pData;

    config->setMaxWorkspaceSize(pParam->nMaxWorkspaceSize);
    if (pParam->bFp16) {
        config->setFlag(BuilderFlag::kFP16);
    }
    
    vProfile[0]->setDimensions("input", OptProfileSelector::kMIN, Dims4(1, pParam->nChannel, pParam->nHeight, pParam->nWidth));
    vProfile[0]->setDimensions("input", OptProfileSelector::kOPT, Dims4(pParam->nMaxBatchSize, pParam->nChannel, pParam->nHeight, pParam->nWidth));
    vProfile[0]->setDimensions("input", OptProfileSelector::kMAX, Dims4(pParam->nMaxBatchSize, pParam->nChannel, pParam->nHeight, pParam->nWidth));
    config->addOptimizationProfile(vProfile[0]);
}

int main(int argc, char** argv) {
    int iDevice = 0;
    if (argc >= 2) iDevice = atoi(argv[1]);
    ck(cudaSetDevice(iDevice));
    cudaDeviceProp prop = {};
    ck(cudaGetDeviceProperties(&prop, iDevice));
    cout << "Using " << prop.name << endl;

    //dlopen("./AppPlugin", RTLD_LAZY);
    // you have to adjust (n,c,h,w) to something like (2,4,1,8) load the engine saved by AppPluginDynamicShape
    //dlopen("./AppPluginDynamicShape", RTLD_LAZY);
    auto trt = unique_ptr<TrtLite>(TrtLiteCreator::Create("/workspace/PanopticFCN/demo/seafood/Engine/panoptic_fcn_7.2.2.trt"));
    trt->PrintInfo();

    if (trt->GetEngine()->isRefittable()) {
        LOG(INFO) << "Engine is refittable. Refitting...";
        //DoRefit(trt.get());
    } else {
        LOG(INFO) << "Engine isn't refittable. Refit is skipped.";
    }

    const int nBatch = 1, nChannel = 3, nHeight = 448, nWidth = 512;
    map<int, Dims> i2shape;
    i2shape.insert(make_pair(0, Dims{4, {nBatch, nChannel, nHeight, nWidth}}));

    vector<void *> vpBuf, vdpBuf;
    vector<IOInfo> vInfo;
    if (trt->GetEngine()->hasImplicitBatchDimension()) {
        vInfo = trt->ConfigIO(nBatch);
    } else {
        vInfo = trt->ConfigIO(i2shape);
    }
    
    for (auto info : vInfo) {
        cout << info.to_string() << endl;
        
        void *pBuf = nullptr;
        pBuf = new float[info.GetNumBytes()];
        vpBuf.push_back(pBuf);

        void *dpBuf = nullptr;
        ck(cudaMalloc(&dpBuf, info.GetNumBytes()));
        vdpBuf.push_back(dpBuf);

        if (info.bInput) {
            fill((float *)pBuf, info.GetNumBytes() / sizeof(float), 1.0f);
            ck(cudaMemcpy(dpBuf, pBuf, info.GetNumBytes(), cudaMemcpyHostToDevice));
        }

    }
    if (trt->GetEngine()->hasImplicitBatchDimension()) {
        trt->Execute(nBatch, vdpBuf);
    } else {
        trt->Execute(i2shape, vdpBuf);
    }
    for (int i = 0; i < vInfo.size(); i++) {
        auto &info = vInfo[i];
        if (info.bInput) {
            continue;
        }
        ck(cudaMemcpy(vpBuf[i], vdpBuf[i], info.GetNumBytes(), cudaMemcpyDeviceToHost));
    }

    print((float *)vpBuf[1], nBatch * nChannel * nHeight, nWidth);    
    
    for (int i = 0; i < vInfo.size(); i++) {
        delete[] (uint8_t *)vpBuf[i];
        ck(cudaFree(vdpBuf[i]));
    }

    return 0;
}