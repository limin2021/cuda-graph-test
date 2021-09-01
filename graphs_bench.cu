// Basic perf test for graph scaling
//
// We'll launch a series of kernels in different graphs and time things.

#include <stdio.h>
#include "timer.h"

#define CUDA(expr) if(expr != cudaSuccess) { printf("Error at line %d: %s\n", __LINE__, cudaGetErrorString(cudaGetLastError())); return -1; }

// Basic delay kernel, timed in ticks
__global__ void delayKernel(unsigned long long ticks) {
    unsigned long long start = clock64();
    while((clock64() - start) < ticks);
}

// Function to read the GPU nanosecond timer in a kernel
__device__ __forceinline__ unsigned long long   __globaltimer() { 
    unsigned long long globaltimer;   
    asm volatile ("mov.u64 %0, %globaltimer;"   : "=l"(globaltimer));   
    return globaltimer; 
}

// The basic kernel: called from a global function.
// It optionally writes out a timestamp
__device__ __forceinline__ void writeTimestamp(unsigned long long *startptr, unsigned long long *endptr) {
    if(startptr != nullptr) {
        *startptr = __globaltimer();
    }
    if(endptr != nullptr) {
        *endptr = __globaltimer();
    }
}

// Need a set of different kernels to launch, to avoid instruction cache re-use
__global__ void k0(unsigned long long *startptr, unsigned long long *endptr) { writeTimestamp(startptr, endptr); }
__global__ void k1(unsigned long long *startptr, unsigned long long *endptr) { writeTimestamp(startptr, endptr); }
__global__ void k2(unsigned long long *startptr, unsigned long long *endptr) { writeTimestamp(startptr, endptr); }
__global__ void k3(unsigned long long *startptr, unsigned long long *endptr) { writeTimestamp(startptr, endptr); }
__global__ void k4(unsigned long long *startptr, unsigned long long *endptr) { writeTimestamp(startptr, endptr); }
__global__ void k5(unsigned long long *startptr, unsigned long long *endptr) { writeTimestamp(startptr, endptr); }
__global__ void k6(unsigned long long *startptr, unsigned long long *endptr) { writeTimestamp(startptr, endptr); }
__global__ void k7(unsigned long long *startptr, unsigned long long *endptr) { writeTimestamp(startptr, endptr); }
__global__ void k8(unsigned long long *startptr, unsigned long long *endptr) { writeTimestamp(startptr, endptr); }
__global__ void k9(unsigned long long *startptr, unsigned long long *endptr) { writeTimestamp(startptr, endptr); }

typedef void (*TimeStampKernel)(unsigned long long*, unsigned long long *);
TimeStampKernel kernels[10] = { k0, k1, k2, k3, k4, k5, k6, k7, k8, k9 };


// Launch a straight-line graph of a given number of nodes, timing all the components
int straightLineGraph(int numnodes, unsigned long long *timeStamps, bool noprint=false) {
    unsigned long long cpuTime;
    cudaStream_t stream;
    cudaEvent_t evStart, evFirst, evEnd;
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    CUDA(cudaStreamCreate(&stream));
    CUDA(cudaEventCreate(&evStart));
    CUDA(cudaEventCreate(&evEnd));
    CUDA(cudaEventCreate(&evFirst));

    // Start by capturing a graph of the appropriate size
    cpuTime = getCurrentTick();
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for(int i=0; i<numnodes; i++) {
        kernels[i%10]<<< 1, 1, 0, stream >>>((i == 0 ? &timeStamps[0] : nullptr), i == numnodes-1 ? &timeStamps[1] : nullptr);
        //kernels[0]<<< 1, 1, 0, stream >>>((i == 0 ? &timeStamps[0] : nullptr), i == numnodes-1 ? &timeStamps[1] : nullptr);
    }
    cudaStreamEndCapture(stream, &graph);
    double createTime = delta_t(cpuTime) * 1e6;

    // Now instantiate it
    cpuTime = getCurrentTick();
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    double instantiateTime = delta_t(cpuTime) * 1e6;

    // First time the graph as if we launched using a stream
    delayKernel<<< 1, 1, 0, stream >>>(100000);      // Delay for 66ms (100k / 1.5GHz)
    cudaEventRecord(evStart, stream);
    cpuTime = getCurrentTick();
    for(int i=0; i<numnodes; i++) {
        kernels[i%10]<<< 1, 1, 0, stream >>>((i == 0 ? &timeStamps[0] : nullptr), i == numnodes-1 ? &timeStamps[1] : nullptr);
        //kernels[0]<<< 1, 1, 0, stream >>>((i == 0 ? &timeStamps[0] : nullptr), i == numnodes-1 ? &timeStamps[1] : nullptr);
    }
    double streamLaunchTime = delta_t(cpuTime) * 1e6;
    cudaEventRecord(evEnd, stream);
    cudaStreamSynchronize(stream);
    float streamRuntime;
    cudaEventElapsedTime(&streamRuntime, evStart, evEnd);
    double streamGlobalTimer = (double)(timeStamps[1] - timeStamps[0]) / 1e3;     // Global timer is in nanoseconds, so convert to microseconds

    // Then we launch graph into the stream multiple times and time its runtime.
    double firstLaunchTime, averageLaunchTime=0.;
    delayKernel<<< 1, 1, 0, stream >>>(100000);      // Delay for 66ms (100k / 1.5GHz)
    cudaEventRecord(evStart, stream);
    int numLaunches = 10;

    for(int launch=0; launch<numLaunches; launch++) {
        cpuTime = getCurrentTick();
        cudaGraphLaunch(graphExec, stream);
        
        double dt = delta_t(cpuTime) * 1e6;
        if(launch == 0) {
            cudaEventRecord(evFirst, stream);       // Timer for just the first kernel
            firstLaunchTime = dt;
        }
        else {
            averageLaunchTime += dt;
        }
    }
    cudaEventRecord(evEnd, stream);
    cudaStreamSynchronize(stream);

    float firstRuntime, averageRuntime;
    cudaEventElapsedTime(&firstRuntime, evStart, evFirst);
    cudaEventElapsedTime(&averageRuntime, evFirst, evEnd);
    double globalTimer = (double)(timeStamps[1] - timeStamps[0]) / 1e3;     // Global timer is in nanoseconds, so convert to microseconds

    averageRuntime /= (numLaunches - 1);
    averageLaunchTime /= (numLaunches - 1);
    streamRuntime *= 1e3;
    firstRuntime *= 1e3;
    averageRuntime *= 1e3;

    //printf("numnodes, createTime, instantiateTime, streamLaunchTime, streamRuntime, streamGlobalTimer, firstLaunchTime, averageLaunchTime, firstRuntime, averageRuntime, globalTimer\n");
    if(!noprint) {
        printf("%d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n", numnodes, createTime, instantiateTime, streamLaunchTime, streamRuntime, streamGlobalTimer, firstLaunchTime, averageLaunchTime, firstRuntime, averageRuntime, globalTimer);
        fflush(stdout);
    }
    return 0;
}


void usage() {
    printf("graphs [numnodes] [numforks] [pattern] [stride]\n");
    printf("\tnumnodes - Number of kernel nodes in each branch of the graph\n");
    printf("\tnumforks - Number of forks which run concurrently\n");
    printf("\t pattern - Structure of graph, default=0 (see below)\n");
    printf("\t  stride - If >0, will sweep from 1->numnodes in step 'stride'\n");
    printf("\n");
    printf("Pattern can be:\n");
    printf("\t0: No interconnect between branches except fork & join at beginning & end\n");
}

int main(int argc, char *argv[]) {
    if(argc < 1) {
        usage();
        return 0;
    }

    int numnodes=20, numforks=1, pattern=0, stride=0;
    if(argc > 1) numnodes = atoi(argv[1]);
    if(argc > 2) numforks = atoi(argv[2]);
    if(argc > 3) pattern = atoi(argv[3]);
    if(argc > 4) stride = atoi(argv[4]);

    if(numnodes == 0) {
        usage();
        return 0;
    }

    CUDA(cudaFree(0));
    unsigned long long *timeStamps;
    CUDA(cudaMallocHost(&timeStamps, 2*sizeof(unsigned long long)));
    timeStamps[0] = timeStamps[1] = 0;

    straightLineGraph(1, timeStamps, true);
    printf("numnodes, createTime, instantiateTime, streamLaunchTime, streamRuntime, streamGlobalTimer, firstLaunchTime, averageLaunchTime, firstRuntime, averageRuntime, globalTimer\n");

    int start = (stride == 0) ? numnodes : 1;
    for(int i=start; i<=numnodes; i+=stride+(stride==0)) {
        straightLineGraph(i, timeStamps);
    }

    return 0;
}

