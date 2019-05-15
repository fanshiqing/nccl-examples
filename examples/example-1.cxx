//
// Example 1: Single Process, Single Thread, Multiple Devices
//

#include <chrono>
#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <stdlib.h>

// __global__ void show(int *in,int i){
//     in[threadIdx.x] = i;
//     printf("%d\n",in[threadIdx.x]);
// }

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

int main(int argc, char *argv[]) {
    /*Get current amounts number of GPU*/
    int nDev = 0;
    cudaGetDeviceCount(&nDev);
    printf("nGPUs = %d\n",nDev);

    /*List GPU Device*/
    int *devs;  
    devs = (int *)malloc( nDev * sizeof(int));
    for (int i = 0; i < nDev; ++i){
        devs[i] = i;
    }

    /*NCCL Init*/
    ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t) * nDev);  
    cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t) * nDev);

    // orflow/core/kernels/pearl_kernels.ccalloc the data
    // int size = 2048 * 100; // 1.2 padding ids size
    // int size = 2048 * 94; // 1.1 ncclBcast ids size
    int size = 0;
    int batch_size = 2048;
    int steps = 94;
    int max_steps = 100;
    int feature_cols = 128;

    int key = 33;
    printf("test key = %d\n", key);
    switch (key) {
      case 11: // ncclBcast(ids)
        size = batch_size * steps;
        break;
      case 12: // padding + ncclAllGather(ids)
        size = batch_size * max_steps;
        break;
      case 31: // ncclReduce(params)
        size = batch_size * steps * feature_cols * (nDev - 1);
        break;
      case 32: // ncclAllReduce(params)
        size = batch_size * steps * feature_cols * (nDev - 1);
        break;
      case 33: // AllGatherV via ncclBcast
        size = batch_size * steps * feature_cols * (nDev - 1);
        break;
      default:
        printf("Unsupported key!\n");
        return -1;
    }
    ncclDataType_t dtype = ncclFloat;

    // allocating and initializing device buffers
    float **sendbuff = (float **) malloc(nDev * sizeof(float *));
    float **recvbuff = (float **) malloc(nDev * sizeof(float *));
    float **recvbuff_allgather = (float **) malloc(nDev * sizeof(float *)); // for allgather test

    // Fake recvcounts for AllGatherV's implementation via ncclBcast
    int* recvcounts = (int *) malloc(nDev * sizeof(int));
    for (int i = 0; i < nDev; ++i) {
        recvcounts[i] = batch_size * steps * feature_cols - i * 10;
    }
    
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
        CUDACHECK(cudaMalloc(recvbuff_allgather + i, size * sizeof(float) * nDev)); // for allgather test
        CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
        CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
        CUDACHECK(cudaMemset(recvbuff_allgather[i], 0, size * sizeof(float) * nDev));
        CUDACHECK(cudaStreamCreate(s + i));
    }

    // initializing NCCL(Note: ncclCommInitAll only for *single process*)
    // for multi-process jobs, please use ncclCommInitRank instead.
    NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

    /*Get GPU status*/
    printf("# Using devices\n");
    for (int g = 0; g < nDev; g++) {
        int cudaDev;
        int rank;
        cudaDeviceProp prop;
        ncclCommCuDevice(comms[g], &cudaDev);
        ncclCommUserRank(comms[g], &rank);

        cudaGetDeviceProperties(&prop, cudaDev);
        printf("#   Rank %2d uses device %2d [0x%02x] %s\n", rank, cudaDev, prop.pciBusID, prop.name);
    }
    printf("\n");

    // GPU Bcast
    printf("Start nccl Bcast...\n");

    auto start = std::chrono::high_resolution_clock::now();
    
    if (key == 11) { 
      /*Option1.1: Bcast Ids to all devices each*/
      for (int root = 0; root < nDev; ++root) {
          NCCLCHECK(ncclGroupStart());
          for (int i = 0; i < nDev; ++i) {
              cudaSetDevice(devs[i]);
              ncclBroadcast((const void *) sendbuff[i],
                            (void *) recvbuff[i],
                            size, ncclInt32, root, comms[i], s[i]); 
          }
          NCCLCHECK(ncclGroupEnd());
      }
    } else if (key == 12) {
      /*Option1.2: Padding [batch_size, max_steps] ids, then allGather(ids)*/
      NCCLCHECK(ncclGroupStart());
      for (int i = 0; i < nDev; ++i) {
          CUDACHECK(cudaSetDevice(devs[i]));
          NCCLCHECK(ncclAllGather((const void *) sendbuff[i],
                                  (void *) recvbuff_allgather[i], size, dtype,
                                  comms[i], s[i]));
      }
      NCCLCHECK(ncclGroupEnd());
    } else if (key == 31) {
      /*Option3.1: Padding [batch_size, max_steps] ids, then allGather(ids)*/
      for (int root = 0; root < nDev; ++root) {
             NCCLCHECK(ncclGroupStart());
             for (int i = 0; i < nDev; ++i) {
                 CUDACHECK(cudaSetDevice(devs[i]));
                 NCCLCHECK(ncclReduce((const void *) sendbuff[i],
                               (void *) recvbuff[i],
                               size, ncclFloat, ncclSum, root, comms[i], s[i])); 
             }
             NCCLCHECK(ncclGroupEnd());
      }
    } else if (key == 32) {
      /*Option3.2: 1.1bcast ids + ncclAllReduce*/
      NCCLCHECK(ncclGroupStart());
      for (int i = 0; i < nDev; ++i) {
          CUDACHECK(cudaSetDevice(devs[i]));
          NCCLCHECK(ncclAllReduce((const void *) sendbuff[i],
                                  (void *) recvbuff[i], size, dtype, ncclSum,
                                  comms[i], s[i]));
      }
      NCCLCHECK(ncclGroupEnd());
    } else if (key == 33) {
      /*Option3.3: AllGatherV via bcast*/
      NCCLCHECK(ncclGroupStart());
      for (int root = 0; root < nDev; ++root) {
          NCCLCHECK(ncclGroupStart());
          for (int i = 0; i < nDev; ++i) {
                 CUDACHECK(cudaSetDevice(devs[i]));
                 NCCLCHECK(ncclBroadcast((const void *) sendbuff[i],
                               (void *) recvbuff[i], recvcounts[i], dtype, root,
                               comms[i], s[i]));
          }
          NCCLCHECK(ncclGroupEnd());
      }
      NCCLCHECK(ncclGroupEnd());
    }

    // synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(s[i]));
    }
    auto delta = std::chrono::high_resolution_clock::now() - start;
    double deltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count();
    printf("Bcast dataset size per device: %.2f MB\n", size*sizeof(dtype)/1000.0/1000.0);
    printf("Bcast time: %f secs\n", deltaSec);
    printf("Bcast done!\n");

    // calling NCCL communication API. Group API is required when
    // using multiple devices per thread
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; ++i) {
        NCCLCHECK(ncclAllReduce((const void *) sendbuff[i],
                                (void *) recvbuff[i], size, ncclFloat, ncclSum,
                                comms[i], s[i]));
    }
    NCCLCHECK(ncclGroupEnd());

    // synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(s[i]));
    }

    // free device buffers
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
    }

    // finalizing NCCL
    for (int i = 0; i < nDev; ++i) {
        ncclCommDestroy(comms[i]);
    }

    printf("Success \n");
    return 0;
}
