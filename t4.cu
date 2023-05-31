#include <iostream>
#include <string>
#include <chrono>
#include <cub/cub.cuh>
#include <cub/block/block_reduce.cuh>
#include "cuda_runtime.h"
using namespace std;

#ifdef NVPROF_
#include </opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda/11.8/targets/x86_64-linux/include/nvtx3/nvToolsExt.h>
#endif


int max_iterations;
double max_err;
int size_arr;
#define at(arr, x, y) (arr[(x)*(n)+(y)])


__global__ void iterate(double *F, double *Fnew, int* size){
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = *size;

    if (j <= 0 || i <= 0 || i >= n - 1 || j >= n - 1) return;
    at(Fnew, i, j) = 0.25 * (at(F, i + 1, j) + at(F, i - 1, j) + at(F, i, j + 1) + at(F, i, j - 1));
}

__global__ void razn_arr(double *F, double *Fnew, double* razn, int* size){
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = *size;

    if (j <= 0 || i <= 0 || i >= n - 1 || j >= n - 1) return;
    at(razn, i, j) = fabs(at(F, i, j) - at(Fnew, i, j));
}

int main(int argc, char *argv[]) {
    cudaSetDevice(2);
    max_err = atof(argv[argc - 3]);
    size_arr = stoi(argv[argc - 2]);
    max_iterations = stoi(argv[argc - 1]);

    auto start = chrono::high_resolution_clock::now();

    double* arr = new double[size_arr * size_arr];
    double *d_new_arr, *d_arr, *razn;

    arr[0] = 10;
    arr[size_arr - 1] = 20;
    arr[(size_arr - 1) * size_arr] = 20;
    arr[size_arr * size_arr - 1] = 30;
    double step = 10.0 / (size_arr - 1);

    for (int i = 1; i < size_arr - 1; i++) {
        arr[i] = 10 + i * step;
        arr[(size_arr - 1) * size_arr + i] = 20 + i * step;
        arr[i * size_arr] = 10 + i * step;
        arr[i * size_arr + size_arr - 1] = 20 + i * step;
    }
    for (int i = 1; i < size_arr - 1; i++) {
        for (int j = 1; j < size_arr - 1; j++) {
            arr[i * size_arr + j] = 20;
        }
    }

    int size_byte = size_arr * size_arr * sizeof(double);
    cudaMalloc(&d_arr, size_byte);
    cudaMalloc(&d_new_arr, size_byte);
    cudaMalloc(&razn, size_byte);
    cudaMemcpy(d_arr, arr, size_byte, cudaMemcpyHostToDevice);
    cudaMemcpy(d_new_arr, arr, size_byte, cudaMemcpyHostToDevice);

    auto elapsed = chrono::high_resolution_clock::now() - start;
    long long msec = chrono::duration_cast<chrono::microseconds>(elapsed).count();
    cout << "initialisation: " << msec << "\n";


    // dim3 threadPerBlock = dim3(32, 32);
    // dim3 blocksPerGrid = dim3((size_arr + 31) / 32, (size_arr + 31) / 32);

    double *d_err;
    cudaMalloc(&d_err, sizeof(double));

    int* size_arr_d;
    cudaMalloc(&size_arr_d, sizeof(int));
    cudaMemcpy(size_arr_d, &size_arr, sizeof(int), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    int iters = 0;
    double err = 1;
    cudaMemcpy(d_err, &err, sizeof(double), cudaMemcpyHostToDevice);

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, razn, d_err, size_arr * size_arr, stream);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);


    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    dim3 threadPerBlock = dim3(32, 32);
    dim3 blocksPerGrid = dim3((size_arr + 31) / 32, (size_arr + 31) / 32);

    for(int i = 0; i < 50; i++){
        iterate<<<blocksPerGrid, threadPerBlock, 0, stream>>>(d_arr, d_new_arr, size_arr_d);
        iterate<<<blocksPerGrid, threadPerBlock, 0, stream>>>(d_new_arr, d_arr, size_arr_d);
    }
    razn_arr<<<blocksPerGrid, threadPerBlock, 0, stream>>>(d_arr, d_new_arr, razn, size_arr_d);

    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

    auto nstart = chrono::high_resolution_clock::now();
    #ifdef NVPROF_
    nvtxRangePush("MainCycle");
    #endif
    while ((err > max_err) && (iters < max_iterations)){
        cudaGraphLaunch(instance, stream);
        cudaDeviceSynchronize();
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, razn, d_err, size_arr * size_arr, stream);
        cudaMemcpy(&err, d_err, sizeof(double), cudaMemcpyDeviceToHost);

        iters = iters + 100;
    }

    #ifdef NVPROF_
    nvtxRangePop();
    #endif

    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);

    auto nelapsed = chrono::high_resolution_clock::now() - nstart;
    msec = chrono::duration_cast<chrono::microseconds>(nelapsed).count();
    cout << "While: " << msec << "\n";

    cout << "Result\n";
    cout << "iterations: " << iters << " error: " << err << "\n";

    delete[] arr;

    return 0;
}