#ifndef __APPLE__  // TODO - maybe consider nvidia support on intel macs?

#include <string.h>
#include <stdlib.h>

#include "gpu_info_tegra.h"

extern "C" {

  void tegra_init(tegra_init_resp_t *resp) {
    tegraReturn_t ret;
    resp->err = NULL;
    resp->th.handle = NULL;
    const int buflen = 256;
    char buf[buflen + 1];
    int i;

    static int deviceHandle;
    cudaError_t cudaInitStatus = cudaSetDevice(0);
    if (cudaInitStatus != cudaSuccess) {
      snprintf(buf, buflen, "Tegra GPU initialization error: %d\n", cudaInitStatus);
      resp->err = strdup(buf);
      return;
    } else {
      resp->th.handle = &deviceHandle;
    }

    int version = 0;
    tegraDriverVersion_t driverVersion;
    driverVersion.major = 0;
    driverVersion.minor = 0;
    driverVersion.err = NULL;

    cudaError_t cudaDriverStatus = cudaDriverGetVersion(&version);
    if (cudaDriverStatus != cudaSuccess) {
      LOG(resp->th.verbose, "tegraSystemGetDriverVersion failed: %d\n", cudaDriverStatus);
    } else {
      driverVersion.major = version / 1000;
      driverVersion.minor = (version - (driverVersion.major * 1000)) / 10;
      LOG(resp->th.verbose, "CUDA driver version: %d-%d\n", driverVersion.major, driverVersion.minor);
    }
  }
}

extern "C" {

  void tegra_check_vram(tegra_handle_t th, mem_info_t *resp) {
    resp->err = NULL;
    tegraMemory_t memInfo = {0};
    tegraReturn_t ret;
    const int buflen = 256;
    char buf[buflen + 1];
    int i;

    if (th.handle == NULL) {
      resp->err = strdup("tegra handle isn't initialized");
      return;
    }
    
    int deviceCount;
    cudaError_t cudaDeviceCountStatus = cudaGetDeviceCount(&deviceCount);
    if (cudaDeviceCountStatus !=TEGRA_SUCCESS) {
      snprintf(buf, buflen, "unable to get device count: %d\n", cudaDeviceCountStatus);
      resp->err = strdup(buf);
      return;
    } else {
      resp->count = (unsigned int)deviceCount;
      *th.tegraDeviceCount = deviceCount;
    }

    resp->total = 0;
    resp->free = 0;
    for (i = 0; i < resp->count; i++) {
      cudaError_t cudaInitStatus = cudaSetDevice(i);
      if (cudaInitStatus != TEGRA_SUCCESS) {
        snprintf(buf, buflen, "unable to initialize device %d: %d\n", i, cudaInitStatus);
        resp->err = strdup(buf);
        return;
      }

      size_t freeMem, totalMem;

      cudaError_t cudaMemInfoStatus = cudaMemGetInfo(&freeMem, &totalMem);
      if (cudaMemInfoStatus != TEGRA_SUCCESS) {
        snprintf(buf, buflen, "unable to retrieve memory info for tegra device %d: %d\n", i, cudaMemInfoStatus);
        resp->err = strdup(buf);
        return;
      } else {
        th.tegraDeviceMemoryInfo->free = freeMem;
        th.tegraDeviceMemoryInfo->total = totalMem;
        th.tegraDeviceMemoryInfo->used = totalMem - freeMem;
        memInfo.total = totalMem;
        memInfo.free = freeMem;
      }

      if (th.verbose) {
        tegraBrandType_t brand = TEGRA_BRAND_UNKNOWN;
        // When in verbose mode, report more information about
        // the card we discover, but don't fail on error

        const char* deviceName = getenv("JETSON_SOC");
        
        if (deviceName != NULL) {
          LOG(th.verbose, "Failed to acquire JETSON_SOC environment variable for device %d\n", i);
        } else {
          snprintf(buf, buflen, deviceName);
          LOG(th.verbose, "[%d] Tegra CUDA device name: %s\n", i, buf);
        }

        // device get serial
        //    JETSON_SERIAL_NUMBER env var
        
        char* deviceSerial = getenv("JETSON_SERIAL_NUMBER");
        
        if (deviceSerial != NULL) {
          LOG(th.verbose, "Failed to acquire JETSON_SOC environment variable for device %d\n", i);
        } else {
          snprintf(buf, buflen, deviceSerial);
          LOG(th.verbose, "[%d] Tegra CUDA S/N: %s\n", i, buf);
        }

        ret = TEGRA_NOT_SUPPORTED;
        if (ret != TEGRA_SUCCESS) {
          LOG(th.verbose, "Retrieve board part number not supported on Tegra devices\n");
        } else {
          LOG(th.verbose, "[%d] CUDA part number: %s\n", i, buf);
        }
        if (ret != TEGRA_SUCCESS) {
          LOG(th.verbose, "GetVbiosVersion not supported on Tegra devices\n");
        } else {
          LOG(th.verbose, "[%d] CUDA vbios version: %s\n", i, buf);
        }
        if (ret != TEGRA_SUCCESS) {
          LOG(th.verbose, "DeviceGetBrand not supported on Tegra devices\n");
        } else {
          LOG(th.verbose, "[%d] CUDA brand: %d\n", i, brand);
        }
      }

      LOG(th.verbose, "[%d] Tegra totalMem %ld\n", i, memInfo.total);
      LOG(th.verbose, "[%d] Tegra usedMem %ld\n", i, memInfo.free);

      resp->total += memInfo.total;
      resp->free += memInfo.free;
    }
  }
}

extern "C" {
  void tegra_compute_capability(tegra_handle_t th, tegra_compute_capability_t *resp) {
    resp->err = NULL;
    resp->major = 0;
    resp->minor = 0;
    int major = 0;
    int minor = 0;
    const int buflen = 256;
    char buf[buflen + 1];
    int i;

    if (th.handle == NULL) {
      resp->err = strdup("tegra handle not initialized");
      return;
    }

    int devices;
    devices = *th.tegraDeviceCount;
    if (devices <= 0) {
      snprintf(buf, buflen, "unable to get device count: %d", devices);
      resp->err = strdup(buf);
      return;
    }

    for (i = 0; i < devices; i++) {
      cudaError_t cudaInitStatus = cudaSetDevice(i);
      if (cudaInitStatus != TEGRA_SUCCESS) {
        snprintf(buf, buflen, "unable to initialize device %d: %d\n", i, cudaInitStatus);
        resp->err = strdup(buf);
        return;
      }

      cudaDeviceProp deviceProp;
      cudaError_t cudaDevicePropertiesStatus = cudaGetDeviceProperties(&deviceProp, i);
      if (cudaDevicePropertiesStatus != TEGRA_SUCCESS) {
        snprintf(buf, buflen, "device compute capability lookup failure %d: %d", i, cudaDevicePropertiesStatus);
        resp->err = strdup(buf);
        return;
      } else {
        major = deviceProp.major;
        minor = deviceProp.minor;
        LOG(th.verbose, "Tegra Compute Capability version: %d.%d\n", major, minor);
      }

      // Report the lowest major.minor we detect as that limits our compatibility
      if (resp->major == 0 || resp->major > major ) {
        resp->major = major;
        resp->minor = minor;
      } else if ( resp->major == major && resp->minor > minor ) {
        resp->minor = minor;
      }
    }
  }
}

#endif  // __APPLE__