#include <stdio.h>
#include <cuda_runtime.h>
#include "gpu_info_tegra.h"

extern "C" {
    nvmlReturn_t nvmlInit_v2(void *) {
        cudaError_t cudaStatus = cudaSetDevice(0); // Sets the first device as active
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(cudaStatus));
            return NVML_UNKNOWN_ERROR; // Map CUDA error to NVML error
        }
        return NVML_SUCCESS;
    }

    nvmlReturn_t nvmlShutdown(void *) {
        cudaError_t cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceReset failed: %s\n", cudaGetErrorString(cudaStatus));
            return NVML_UNKNOWN_ERROR;
        }
        return NVML_SUCCESS;
    }

    nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device) {
        // In CUDA Runtime API, devices are typically referred to by their index
        *device = (nvmlDevice_t)index; // Directly store the index as the "handle"
        return NVML_SUCCESS;
    }

    nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory) {
        size_t freeMem, totalMem;
        cudaError_t cudaStatus = cudaMemGetInfo(&freeMem, &totalMem);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemGetInfo failed: %s\n", cudaGetErrorString(cudaStatus));
            return NVML_UNKNOWN_ERROR;
        }
        memory->total = totalMem;
        memory->free = freeMem;
        memory->used = totalMem - freeMem;
        return NVML_SUCCESS;
    }

    nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount) {
        int count;
        cudaError_t cudaStatus = cudaGetDeviceCount(&count);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(cudaStatus));
            return NVML_UNKNOWN_ERROR;
        }
        *deviceCount = (unsigned int)count;
        return NVML_SUCCESS;
    }

    nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int *major, int *minor) {
        cudaDeviceProp deviceProp;
        int device_handle = 0;
        cudaError_t cudaStatus = cudaGetDeviceProperties(&deviceProp, device_handle);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(cudaStatus));
            return NVML_UNKNOWN_ERROR;
        }
        *major = deviceProp.major;
        *minor = deviceProp.minor;
        return NVML_SUCCESS;
    }

}