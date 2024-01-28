#include <stdio.h>
#include <string.h>
#include "gpu_info_tegra.h"

int main() {

    nvmlMemory_t *memory;
    size_t freeMem, totalMem;

    // cudaDeviceCount -> nvmlDeviceGetCount_v2
    int deviceCount; // 

    // cudaGetDeviceProperties -> nvmlDeviceGetCudaComputeCapability
    int major = 0;
    int minor = 0;

    // Execute test functions
    // cudaSetDevice
    cudaError_t cudaInitStatus = cudaSetDevice(0);

    // cudaDeviceReset
    cudaError_t cudaResetStatus = cudaDeviceReset();

    // nvmlDeviceGetHandleByIndex -- Directly store the index as the "handle". Not directly used.
    // handle_device = (nvmlDevice_t)index;

    // cudaMemGetInfo
    cudaError_t cudaMemInfoStatus = cudaMemGetInfo(&freeMem, &totalMem);
    cudaError_t cudaDeviceCountStatus = cudaGetDeviceCount(&deviceCount);

    // cudaGetDeviceProperties -> nvmlDevicesComputeCapability
    cudaDeviceProp deviceProp;
    int device_handle = 0;
    cudaError_t cudaDevicePropertiesStatus = cudaGetDeviceProperties(&deviceProp, device_handle);
    
    major = deviceProp.major;
    minor = deviceProp.minor;

    int total_mb = totalMem / (1024 * 1024);
    int free_mb = freeMem / (1024 * 1024);
    int used_mb = (totalMem - freeMem) / (1024 * 1024);
    int usedMem = (totalMem - freeMem);

    printf("Device Number: %d || nvmlInit_v2() and nvmlShutDown() good.\n", device_handle);
    printf("  Memory info:\n");
    printf("    Total: %d (%d MB)\n", totalMem, total_mb);
    printf("    Used:  %d (%d MB)\n", usedMem, used_mb);
    printf("    Free:  %d (%d MB)\n", freeMem, free_mb);
    printf("\n");
    printf("  Device Count: %d\n", deviceCount);
    printf("  CUDA Compute Capability: %d.%d\n", major, minor);

}