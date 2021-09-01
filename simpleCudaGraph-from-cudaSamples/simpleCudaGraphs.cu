/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <vector>
#include <iostream>

namespace cg = cooperative_groups;

//#define REDUCE

#define THREADS_PER_BLOCK 512
#define GRAPH_LAUNCH_ITERATIONS 100


#define NSTEP 10000
#define NKERNEL 20
#define N 500000 // tuned such that kernel takes a few microseconds

typedef struct callBackData {
  const char *fn_name;
  double *data;
} callBackData_t;


__global__ void shortKernel(float *out_d, float *in_d){
#if 0
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	  if(idx < N){
	    out_d[idx] = 1.23 * in_d[idx];
	  }
#endif
}


__global__ void reduce(float *inputVec, double *outputVec, size_t inputSize,
                       size_t outputSize) {
  __shared__ double tmp[THREADS_PER_BLOCK];

  cg::thread_block cta = cg::this_thread_block();
  size_t globaltid = blockIdx.x * blockDim.x + threadIdx.x;

  double temp_sum = 0.0;
  for (int i = globaltid; i < inputSize; i += gridDim.x * blockDim.x) {
    temp_sum += (double)inputVec[i];
  }
  tmp[cta.thread_rank()] = temp_sum;

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  double beta = temp_sum;
  double temp;

  for (int i = tile32.size() / 2; i > 0; i >>= 1) {
    if (tile32.thread_rank() < i) {
      temp = tmp[cta.thread_rank() + i];
      beta += temp;
      tmp[cta.thread_rank()] = beta;
    }
    cg::sync(tile32);
  }
  cg::sync(cta);

  if (cta.thread_rank() == 0 && blockIdx.x < outputSize) {
    beta = 0.0;
    for (int i = 0; i < cta.size(); i += tile32.size()) {
      beta += tmp[i];
    }
    outputVec[blockIdx.x] = beta;
  }
}

__global__ void reduceFinal(double *inputVec, double *result,
                            size_t inputSize) {
  __shared__ double tmp[THREADS_PER_BLOCK];

  cg::thread_block cta = cg::this_thread_block();
  size_t globaltid = blockIdx.x * blockDim.x + threadIdx.x;

  double temp_sum = 0.0;
  for (int i = globaltid; i < inputSize; i += gridDim.x * blockDim.x) {
    temp_sum += (double)inputVec[i];
  }
  tmp[cta.thread_rank()] = temp_sum;

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  // do reduction in shared mem
  if ((blockDim.x >= 512) && (cta.thread_rank() < 256)) {
    tmp[cta.thread_rank()] = temp_sum = temp_sum + tmp[cta.thread_rank() + 256];
  }

  cg::sync(cta);

  if ((blockDim.x >= 256) && (cta.thread_rank() < 128)) {
    tmp[cta.thread_rank()] = temp_sum = temp_sum + tmp[cta.thread_rank() + 128];
  }

  cg::sync(cta);

  if ((blockDim.x >= 128) && (cta.thread_rank() < 64)) {
    tmp[cta.thread_rank()] = temp_sum = temp_sum + tmp[cta.thread_rank() + 64];
  }

  cg::sync(cta);

  if (cta.thread_rank() < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (blockDim.x >= 64) temp_sum += tmp[cta.thread_rank() + 32];
    // Reduce final warp using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      temp_sum += tile32.shfl_down(temp_sum, offset);
    }
  }
  // write result for this block to global mem
  if (cta.thread_rank() == 0) result[0] = temp_sum;
}

void init_input(float *a, size_t size) {
  for (size_t i = 0; i < size; i++) a[i] = (rand() & 0xFF) / (float)RAND_MAX;
}

void CUDART_CB myHostNodeCallback(void *data) {
  // Check status of GPU after stream operations are done
  callBackData_t *tmp = (callBackData_t *)(data);
  // checkCudaErrors(tmp->status);

  double *result = (double *)(tmp->data);
  char *function = (char *)(tmp->fn_name);
  printf("[%s] Host callback final reduced sum = %lf\n", function, *result);
  *result = 0.0;  // reset the result
}

void cudaGraphsManual(float *inputVec_h, float *inputVec_d, double *outputVec_d,
                      double *result_d, size_t inputSize, size_t numOfBlocks) {
  cudaStream_t streamForGraph;
  cudaGraph_t graph;
  std::vector<cudaGraphNode_t> nodeDependencies;
  cudaGraphNode_t memcpyNode, kernelNode, memsetNode;
  double result_h = 0.0;

  checkCudaErrors(cudaStreamCreate(&streamForGraph));

  cudaKernelNodeParams kernelNodeParams = {0};
  cudaMemcpy3DParms memcpyParams = {0};
  cudaMemsetParams memsetParams = {0};

  memcpyParams.srcArray = NULL;
  memcpyParams.srcPos = make_cudaPos(0, 0, 0);
  memcpyParams.srcPtr =
      make_cudaPitchedPtr(inputVec_h, sizeof(float) * inputSize, inputSize, 1);
  memcpyParams.dstArray = NULL;
  memcpyParams.dstPos = make_cudaPos(0, 0, 0);
  memcpyParams.dstPtr =
      make_cudaPitchedPtr(inputVec_d, sizeof(float) * inputSize, inputSize, 1);
  memcpyParams.extent = make_cudaExtent(sizeof(float) * inputSize, 1, 1);
  memcpyParams.kind = cudaMemcpyHostToDevice;

  memsetParams.dst = (void *)outputVec_d;
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(float);  // elementSize can be max 4 bytes
  memsetParams.width = numOfBlocks * 2;
  memsetParams.height = 1;

  checkCudaErrors(cudaGraphCreate(&graph, 0));
  checkCudaErrors(
      cudaGraphAddMemcpyNode(&memcpyNode, graph, NULL, 0, &memcpyParams));
  checkCudaErrors(
      cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams));

  nodeDependencies.push_back(memsetNode);
  nodeDependencies.push_back(memcpyNode);

  void *kernelArgs[4] = {(void *)&inputVec_d, (void *)&outputVec_d, &inputSize,
                         &numOfBlocks};

  kernelNodeParams.func = (void *)reduce;
  kernelNodeParams.gridDim = dim3(numOfBlocks, 1, 1);
  kernelNodeParams.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = (void **)kernelArgs;
  kernelNodeParams.extra = NULL;

  checkCudaErrors(
      cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), &kernelNodeParams));

  nodeDependencies.clear();
  nodeDependencies.push_back(kernelNode);

  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = result_d;
  memsetParams.value = 0;
  memsetParams.elementSize = sizeof(float);
  memsetParams.width = 2;
  memsetParams.height = 1;
  checkCudaErrors(
      cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams));

  nodeDependencies.push_back(memsetNode);

  memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
  kernelNodeParams.func = (void *)reduceFinal;
  kernelNodeParams.gridDim = dim3(1, 1, 1);
  kernelNodeParams.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  void *kernelArgs2[3] = {(void *)&outputVec_d, (void *)&result_d,
                          &numOfBlocks};
  kernelNodeParams.kernelParams = kernelArgs2;
  kernelNodeParams.extra = NULL;

  checkCudaErrors(
      cudaGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), &kernelNodeParams));
  nodeDependencies.clear();
  nodeDependencies.push_back(kernelNode);

  memset(&memcpyParams, 0, sizeof(memcpyParams));

  memcpyParams.srcArray = NULL;
  memcpyParams.srcPos = make_cudaPos(0, 0, 0);
  memcpyParams.srcPtr = make_cudaPitchedPtr(result_d, sizeof(double), 1, 1);
  memcpyParams.dstArray = NULL;
  memcpyParams.dstPos = make_cudaPos(0, 0, 0);
  memcpyParams.dstPtr = make_cudaPitchedPtr(&result_h, sizeof(double), 1, 1);
  memcpyParams.extent = make_cudaExtent(sizeof(double), 1, 1);
  memcpyParams.kind = cudaMemcpyDeviceToHost;
  checkCudaErrors(
      cudaGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), &memcpyParams));
  nodeDependencies.clear();
  nodeDependencies.push_back(memcpyNode);

  cudaGraphNode_t hostNode;
  cudaHostNodeParams hostParams = {0};
  hostParams.fn = myHostNodeCallback;
  callBackData_t hostFnData;
  hostFnData.data = &result_h;
  hostFnData.fn_name = "cudaGraphsManual";
  hostParams.userData = &hostFnData;

  checkCudaErrors(cudaGraphAddHostNode(&hostNode, graph,
                                       nodeDependencies.data(),
                                       nodeDependencies.size(), &hostParams));

  cudaGraphNode_t *nodes = NULL;
  size_t numNodes = 0;
  checkCudaErrors(cudaGraphGetNodes(graph, nodes, &numNodes));
  printf("\nNum of nodes in the graph created manually = %zu\n", numNodes);

  cudaGraphExec_t graphExec;
  checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

  cudaGraph_t clonedGraph;
  cudaGraphExec_t clonedGraphExec;
  checkCudaErrors(cudaGraphClone(&clonedGraph, graph));
  checkCudaErrors(
      cudaGraphInstantiate(&clonedGraphExec, clonedGraph, NULL, NULL, 0));

  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    checkCudaErrors(cudaGraphLaunch(graphExec, streamForGraph));
  }

  checkCudaErrors(cudaStreamSynchronize(streamForGraph));

  printf("Cloned Graph Output.. \n");
  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    checkCudaErrors(cudaGraphLaunch(clonedGraphExec, streamForGraph));
  }
  checkCudaErrors(cudaStreamSynchronize(streamForGraph));

  checkCudaErrors(cudaGraphExecDestroy(graphExec));
  checkCudaErrors(cudaGraphExecDestroy(clonedGraphExec));
  checkCudaErrors(cudaGraphDestroy(graph));
  checkCudaErrors(cudaGraphDestroy(clonedGraph));
  checkCudaErrors(cudaStreamDestroy(streamForGraph));
}

//limin-todo
void simpleRun(float *inputVec_h, float *inputVec_d,
               double *outputVec_d, double *result_d,
               size_t inputSize, size_t numOfBlocks) {
  cudaStream_t stream1;
  //cudaEvent_t forkStreamEvent, memsetEvent1, memsetEvent2;
  double result_h = 0.0;

  checkCudaErrors(cudaStreamCreate(&stream1));
#if 0
  checkCudaErrors(cudaStreamCreate(&stream2));
  checkCudaErrors(cudaStreamCreate(&stream3));
  checkCudaErrors(cudaStreamCreate(&streamForGraph));

  checkCudaErrors(cudaEventCreate(&forkStreamEvent));
  checkCudaErrors(cudaEventCreate(&memsetEvent1));
  checkCudaErrors(cudaEventCreate(&memsetEvent2));
#endif

  cudaEvent_t e_start, e_stop;
  cudaEventCreate(&e_start);
  cudaEventCreate(&e_stop);
  cudaEventRecord(e_start, stream1);
  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
  checkCudaErrors(cudaMemcpyAsync(inputVec_d, inputVec_h,
                                  sizeof(float) * inputSize, cudaMemcpyDefault,
                                  stream1));
  checkCudaErrors(
      cudaMemsetAsync(outputVec_d, 0, sizeof(double) * numOfBlocks, stream1));

  checkCudaErrors(cudaMemsetAsync(result_d, 0, sizeof(double), stream1));

  reduce<<<numOfBlocks, THREADS_PER_BLOCK, 0, stream1>>>(
      inputVec_d, outputVec_d, inputSize, numOfBlocks);

  reduceFinal<<<1, THREADS_PER_BLOCK, 0, stream1>>>(outputVec_d, result_d,
                                                    numOfBlocks);
  checkCudaErrors(cudaMemcpyAsync(&result_h, result_d, sizeof(double),
                                  cudaMemcpyDefault, stream1));
 }
  cudaEventRecord(e_stop, stream1); 
  cudaEventSynchronize(e_stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
  std::cout << "limin: simpleRun time is " << elapsedTime << std::endl; 

}


//limin-todo
void simpleStreamRun(float *inputVec_h, float *inputVec_d,
               double *outputVec_d, double *result_d,
               size_t inputSize, size_t numOfBlocks) {
  cudaStream_t stream1, stream2, stream3, streamForGraph;
  cudaEvent_t forkStreamEvent, memsetEvent1, memsetEvent2;
  double result_h = 0.0;

  checkCudaErrors(cudaStreamCreate(&stream1));
  checkCudaErrors(cudaStreamCreate(&stream2));
  checkCudaErrors(cudaStreamCreate(&stream3));
  checkCudaErrors(cudaStreamCreate(&streamForGraph));

  checkCudaErrors(cudaEventCreate(&forkStreamEvent));
  checkCudaErrors(cudaEventCreate(&memsetEvent1));
  checkCudaErrors(cudaEventCreate(&memsetEvent2));

  cudaEvent_t e_start, e_stop;
  cudaEventCreate(&e_start);
  cudaEventCreate(&e_stop);
  cudaEventRecord(e_start, stream1);
  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
  checkCudaErrors(cudaEventRecord(forkStreamEvent, stream1));
  checkCudaErrors(cudaStreamWaitEvent(stream2, forkStreamEvent, 0));
  checkCudaErrors(cudaStreamWaitEvent(stream3, forkStreamEvent, 0));

  checkCudaErrors(cudaMemcpyAsync(inputVec_d, inputVec_h,
                                  sizeof(float) * inputSize, cudaMemcpyDefault,
                                  stream1));

  checkCudaErrors(
      cudaMemsetAsync(outputVec_d, 0, sizeof(double) * numOfBlocks, stream2));

  checkCudaErrors(cudaEventRecord(memsetEvent1, stream2));

  checkCudaErrors(cudaMemsetAsync(result_d, 0, sizeof(double), stream3));
  checkCudaErrors(cudaEventRecord(memsetEvent2, stream3));

  checkCudaErrors(cudaStreamWaitEvent(stream1, memsetEvent1, 0));

  reduce<<<numOfBlocks, THREADS_PER_BLOCK, 0, stream1>>>(
      inputVec_d, outputVec_d, inputSize, numOfBlocks);

  checkCudaErrors(cudaStreamWaitEvent(stream1, memsetEvent2, 0));

  reduceFinal<<<1, THREADS_PER_BLOCK, 0, stream1>>>(outputVec_d, result_d,
                                                    numOfBlocks);
  checkCudaErrors(cudaMemcpyAsync(&result_h, result_d, sizeof(double),
                                  cudaMemcpyDefault, stream1));
 }
  //cudaEventRecord(e_stop, streamForGraph); 
  cudaEventRecord(e_stop, stream1); 
  cudaEventSynchronize(e_stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
  std::cout << "limin: simpleStreamRun time is " << elapsedTime << std::endl; 
 
  checkCudaErrors(cudaStreamSynchronize(streamForGraph));

}



//limin-todo:
void cudaGraphsUsingStreamCapture(float *inputVec_h, float *inputVec_d,
                                  double *outputVec_d, double *result_d,
                                  size_t inputSize, size_t numOfBlocks) {
  cudaStream_t stream1, stream2, stream3, streamForGraph;
  cudaEvent_t forkStreamEvent, memsetEvent1, memsetEvent2;
  cudaGraph_t graph;
  double result_h = 0.0;

  checkCudaErrors(cudaStreamCreate(&stream1));
  checkCudaErrors(cudaStreamCreate(&stream2));
  checkCudaErrors(cudaStreamCreate(&stream3));
  checkCudaErrors(cudaStreamCreate(&streamForGraph));

  checkCudaErrors(cudaEventCreate(&forkStreamEvent));
  checkCudaErrors(cudaEventCreate(&memsetEvent1));
  checkCudaErrors(cudaEventCreate(&memsetEvent2));

  checkCudaErrors(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));

  checkCudaErrors(cudaEventRecord(forkStreamEvent, stream1));
  checkCudaErrors(cudaStreamWaitEvent(stream2, forkStreamEvent, 0));
  checkCudaErrors(cudaStreamWaitEvent(stream3, forkStreamEvent, 0));

  checkCudaErrors(cudaMemcpyAsync(inputVec_d, inputVec_h,
                                  sizeof(float) * inputSize, cudaMemcpyDefault,
                                  stream1));

  checkCudaErrors(
      cudaMemsetAsync(outputVec_d, 0, sizeof(double) * numOfBlocks, stream2));

  checkCudaErrors(cudaEventRecord(memsetEvent1, stream2));

  checkCudaErrors(cudaMemsetAsync(result_d, 0, sizeof(double), stream3));
  checkCudaErrors(cudaEventRecord(memsetEvent2, stream3));

  checkCudaErrors(cudaStreamWaitEvent(stream1, memsetEvent1, 0));

  reduce<<<numOfBlocks, THREADS_PER_BLOCK, 0, stream1>>>(
      inputVec_d, outputVec_d, inputSize, numOfBlocks);

  checkCudaErrors(cudaStreamWaitEvent(stream1, memsetEvent2, 0));

  reduceFinal<<<1, THREADS_PER_BLOCK, 0, stream1>>>(outputVec_d, result_d,
                                                    numOfBlocks);
  checkCudaErrors(cudaMemcpyAsync(&result_h, result_d, sizeof(double),
                                  cudaMemcpyDefault, stream1));

  callBackData_t hostFnData = {0};
  hostFnData.data = &result_h;
  hostFnData.fn_name = "cudaGraphsUsingStreamCapture";
  cudaHostFn_t fn = myHostNodeCallback;
  checkCudaErrors(cudaLaunchHostFunc(stream1, fn, &hostFnData));
  checkCudaErrors(cudaStreamEndCapture(stream1, &graph));

  cudaGraphNode_t *nodes = NULL;
  size_t numNodes = 0;
  checkCudaErrors(cudaGraphGetNodes(graph, nodes, &numNodes));
  printf("\nNum of nodes in the graph created using stream capture API = %zu\n",
         numNodes);

  cudaEvent_t e_start, e_stop;
  cudaEventCreate(&e_start);
  cudaEventCreate(&e_stop);

  cudaGraphExec_t graphExec;
  checkCudaErrors(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

  cudaGraph_t clonedGraph;
  cudaGraphExec_t clonedGraphExec;
  checkCudaErrors(cudaGraphClone(&clonedGraph, graph));
  checkCudaErrors(
      cudaGraphInstantiate(&clonedGraphExec, clonedGraph, NULL, NULL, 0));
  
  cudaEventRecord(e_start, 0);
  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    checkCudaErrors(cudaGraphLaunch(graphExec, streamForGraph));
  }
  cudaEventRecord(e_stop, streamForGraph); 
  cudaEventSynchronize(e_stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
  std::cout << "limin: cuda graph time is " << elapsedTime << std::endl; 
 
  checkCudaErrors(cudaStreamSynchronize(streamForGraph));
#if 0
  printf("Cloned Graph Output.. \n");
  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    checkCudaErrors(cudaGraphLaunch(clonedGraphExec, streamForGraph));
  }

  checkCudaErrors(cudaStreamSynchronize(streamForGraph));
#endif

  checkCudaErrors(cudaGraphExecDestroy(graphExec));
  checkCudaErrors(cudaGraphExecDestroy(clonedGraphExec));
  checkCudaErrors(cudaGraphDestroy(graph));
  checkCudaErrors(cudaGraphDestroy(clonedGraph));
  checkCudaErrors(cudaStreamDestroy(stream1));
  checkCudaErrors(cudaStreamDestroy(stream2));
  checkCudaErrors(cudaStreamDestroy(streamForGraph));
}

int test_shortKernel_ver1(float* d_in, float* d_out){
  int threads = 512;
  int blocks = (N + threads - 1)/threads;
#if 0 
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
#endif

  cudaEvent_t e_start, e_stop;
  cudaEventCreate(&e_start);
  cudaEventCreate(&e_stop);
  cudaEventRecord(e_start, 0);
  for(int istep=0; istep<NSTEP; istep++){
	  for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
		shortKernel<<<blocks, threads, 0, 0>>>(d_out, d_in);
		cudaStreamSynchronize(0);
	  }
  }
  cudaEventRecord(e_stop, 0); 
  cudaEventSynchronize(e_stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
  std::cout << "limin: shortkernel ver1 time is " << elapsedTime << std::endl; 
return 0;
}

//
int test_shortKernel_ver2(float* d_in, float* d_out){
  int threads = 512;
  int blocks = (N + threads - 1)/threads;
#if 0
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
#endif

  cudaEvent_t e_start, e_stop;
  cudaEventCreate(&e_start);
  cudaEventCreate(&e_stop);
  cudaEventRecord(e_start, 0);
  for(int istep=0; istep<NSTEP; istep++){
	  for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
		shortKernel<<<blocks, threads, 0, 0>>>(d_out, d_in);
	  }
	  //cudaStreamSynchronize(0);
  }
  cudaStreamSynchronize(0);
  cudaEventRecord(e_stop, 0); 
  cudaEventSynchronize(e_stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
  std::cout << "limin: shortkernel ver2 time is " << elapsedTime << std::endl; 

return 0;
}

//cuda-graph
int test_shortKernel_ver3(float* d_in, float* d_out){
  int threads = 512;
  int blocks = (N + threads - 1)/threads;

  bool graphCreated=false;
  cudaGraph_t graph;
  cudaGraphExec_t instance;
#if 1 
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
#endif

  cudaEvent_t e_start, e_stop;
  cudaEventCreate(&e_start);
  cudaEventCreate(&e_stop);
  cudaEventRecord(e_start, stream);
  
  for(int istep=0; istep<NSTEP; istep++){
	if(!graphCreated){
		cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
		for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
			shortKernel<<<blocks, threads, 0, stream>>>(d_out, d_in);
		}
		cudaStreamEndCapture(stream, &graph);
		cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
		graphCreated=true;
	}
    cudaGraphLaunch(instance, stream);
    //cudaStreamSynchronize(stream);
  }

  cudaEventRecord(e_stop, stream); 
  cudaEventSynchronize(e_stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
  std::cout << "limin: shortkernel ver3 time is " << elapsedTime << std::endl; 
  
  checkCudaErrors(cudaStreamDestroy(stream));
  checkCudaErrors(cudaGraphExecDestroy(instance));
  checkCudaErrors(cudaGraphDestroy(graph));
 return 0;
}


int test_shortKernel_ver4(float* d_in, float* d_out){
  int threads = 512;
  int blocks = (N + threads - 1)/threads;

  //bool graphCreated=false;
  cudaGraph_t graph;
  cudaGraphExec_t instance;
#if 1 
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
#endif
  
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
     shortKernel<<<blocks, threads, 0, stream>>>(d_out, d_in);
  }
  cudaStreamEndCapture(stream, &graph);
  cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
  //graphCreated=true;

  cudaEvent_t e_start, e_stop;
  cudaEventCreate(&e_start);
  cudaEventCreate(&e_stop);
  cudaEventRecord(e_start, stream);
  
  for(int istep=0; istep<NSTEP; istep++){
    cudaGraphLaunch(instance, stream);
    //cudaStreamSynchronize(stream);
  }

  cudaEventRecord(e_stop, stream); 
  cudaEventSynchronize(e_stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
  std::cout << "limin: shortkernel ver4 time (only cudaGraphLaunch time) is " << elapsedTime << std::endl; 
  
  checkCudaErrors(cudaStreamDestroy(stream));
  checkCudaErrors(cudaGraphExecDestroy(instance));
  checkCudaErrors(cudaGraphDestroy(graph));
 return 0;
}


int test_shortKernel_driver(){
  int size = N;
  float *inputVec_d = NULL, *inputVec_h = NULL;
  float *outputVec_d = NULL, *outputVec_h = NULL;

  checkCudaErrors(cudaMallocHost(&inputVec_h, sizeof(float) * size));
  checkCudaErrors(cudaMallocHost(&outputVec_h, sizeof(float) * size));
  
  checkCudaErrors(cudaMalloc(&inputVec_d, sizeof(float) * size));
  checkCudaErrors(cudaMalloc(&outputVec_d, sizeof(float) * size));
  init_input(inputVec_h, size);
  
  checkCudaErrors(cudaMemcpyAsync(inputVec_d, inputVec_h,
                                  sizeof(float) * size, cudaMemcpyHostToDevice, 0));
                                  //sizeof(float) * size, cudaMemcpyDefault, 0));
  //checkCudaErrors(cudaMemsetAsync(outputVec_d, 0, sizeof(float)*size, 0));
#if 1 
  //limin-todo:
  // test_shortKernel_ver1(inputVec_d, outputVec_d);
  // test_shortKernel_ver1(inputVec_d, outputVec_d);
  //test_shortKernel_ver2(inputVec_d, outputVec_d);
  //test_shortKernel_ver3(inputVec_d, outputVec_d);
  test_shortKernel_ver4(inputVec_d, outputVec_d);
#endif

#if 1 
  checkCudaErrors(cudaMemcpyAsync(outputVec_h, outputVec_d, sizeof(float)*size,
                                  cudaMemcpyDeviceToHost, 0));
                                  //cudaMemcpyDefault, 0));

  cudaStreamSynchronize(0);
#endif

  checkCudaErrors(cudaFreeHost(inputVec_h));
  checkCudaErrors(cudaFreeHost(outputVec_h));
  
  checkCudaErrors(cudaFree(inputVec_d));
  checkCudaErrors(cudaFree(outputVec_d));
  
return 0;
}





int main(int argc, char **argv) {
#ifdef REDUCE
  size_t size = 1 << 24;  // number of elements to reduce
  size_t maxBlocks = 512;

  // This will pick the best possible CUDA capable device
  int devID = findCudaDevice(argc, (const char **)argv);

  printf("%zu elements\n", size);
  printf("threads per block  = %d\n", THREADS_PER_BLOCK);
  printf("Graph Launch iterations = %d\n", GRAPH_LAUNCH_ITERATIONS);

  float *inputVec_d = NULL, *inputVec_h = NULL;
  double *outputVec_d = NULL, *result_d;

  checkCudaErrors(cudaMallocHost(&inputVec_h, sizeof(float) * size));
  checkCudaErrors(cudaMalloc(&inputVec_d, sizeof(float) * size));
  checkCudaErrors(cudaMalloc(&outputVec_d, sizeof(double) * maxBlocks));
  checkCudaErrors(cudaMalloc(&result_d, sizeof(double)));

  init_input(inputVec_h, size);
#if 0
  cudaGraphsManual(inputVec_h, inputVec_d, outputVec_d, result_d, size,
                   maxBlocks);
#endif
  cudaGraphsUsingStreamCapture(inputVec_h, inputVec_d, outputVec_d, result_d,
                               size, maxBlocks);
  //simpleStreamRun(inputVec_h, inputVec_d, outputVec_d, result_d, size, maxBlocks);
  //simpleRun(inputVec_h, inputVec_d, outputVec_d, result_d, size, maxBlocks);

  checkCudaErrors(cudaFree(inputVec_d));
  checkCudaErrors(cudaFree(outputVec_d));
  checkCudaErrors(cudaFree(result_d));
  checkCudaErrors(cudaFreeHost(inputVec_h));
#else
  test_shortKernel_driver();
#endif
  return EXIT_SUCCESS;
}
