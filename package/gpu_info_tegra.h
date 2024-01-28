// Just enough typedef's to dlopen/dlsym for memory information
typedef enum nvmlReturn_enum {
  NVML_SUCCESS = 0,
  NVML_UNKNOWN_ERROR = 1,
  // Other values omitted for now...
} nvmlReturn_t;
typedef void *nvmlDevice_t;  // Code to get device handle needs updated, I think setting it to int messed it up.
typedef struct nvmlMemory_st {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
} nvmlMemory_t;

typedef struct cuda_handle {
  void *handle;
  nvmlReturn_t (*initFn)(void);
  nvmlReturn_t (*shutdownFn)(void);
  nvmlReturn_t (*getHandle)(int, nvmlDevice_t *);
  nvmlReturn_t (*getMemInfo)(nvmlDevice_t, nvmlMemory_t *);
  nvmlReturn_t (*getCount)(int *);
  nvmlReturn_t (*getComputeCapability)(nvmlDevice_t, int* major, int* minor);
} cuda_handle_t;

typedef struct cuda_init_resp {
  char *err;  // If err is non-null handle is invalid
  cuda_handle_t ch;
} cuda_init_resp_t;

typedef struct cuda_compute_capability {
  char *err;
  int major;
  int minor;
} cuda_compute_capability_t;