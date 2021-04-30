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

#include "SeafoodPlugin.h"
#include "cuda_fp16.h"
#include <chrono>
#include <thread>

template<typename T>
__global__ void Seafood(T *pDst, T *pSrc, int n) {
    //多block多thread
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = xIndex + yIndex;

    while(offset < n){
        if(pSrc[offset] < 0){
            pDst[offset] = 0;
        }
        else{
            pDst[offset] = pSrc[offset];
        }
    }
    
}

int SeafoodPlugin::enqueue(int nBatch, const void * const *inputs, void **outputs, void* workspace, cudaStream_t stream) {
    int n = nBatch;
    for (int i = 0; i < m.inputDim.nbDims; i++) {
        n *= m.inputDim.d[i];
    }
    printf("n=%d, nBatch=%d\n", n, nBatch);
        //定义block thread范围
    Seafood<<<1,256>>>((float *)outputs[0], (float *)inputs[0], n);
    return 0;
}

REGISTER_TENSORRT_PLUGIN(SeafoodPluginCreator);
