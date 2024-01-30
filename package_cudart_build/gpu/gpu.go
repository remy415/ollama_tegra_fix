//go:build linux || windows

package gpu

/*
#cgo linux LDFLAGS: -lrt -lpthread -ldl -lstdc++ -lm
#cgo windows LDFLAGS: -lpthread

#include "gpu_info.h"

*/
import "C"
import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"unsafe"

	"github.com/pbnjay/memory"
)

type handles struct {
	cuda *C.cuda_handle_t
	rocm *C.rocm_handle_t
	tegra *C.tegra_handle_t
}

var gpuMutex sync.Mutex
var gpuHandles *handles = nil

// With our current CUDA compile flags, older than 5.0 will not work properly
var CudaComputeMin = [2]C.int{5, 0}

// Possible locations for the nvidia-ml library
var CudaLinuxGlobs = []string{
	"/usr/local/cuda/lib64/libnvidia-ml.so*",
	"/usr/lib/x86_64-linux-gnu/nvidia/current/libnvidia-ml.so*",
	"/usr/lib/x86_64-linux-gnu/libnvidia-ml.so*",
	"/usr/lib/wsl/lib/libnvidia-ml.so*",
	"/usr/lib/wsl/drivers/*/libnvidia-ml.so*",
	"/opt/cuda/lib64/libnvidia-ml.so*",
	"/usr/lib*/libnvidia-ml.so*",
	"/usr/local/lib*/libnvidia-ml.so*",
	"/usr/lib/aarch64-linux-gnu/nvidia/current/libnvidia-ml.so*",
	"/usr/lib/aarch64-linux-gnu/libnvidia-ml.so*",

	// TODO: are these stubs ever valid?
	"/opt/cuda/targets/x86_64-linux/lib/stubs/libnvidia-ml.so*",
}

var CudaWindowsGlobs = []string{
	"c:\\Windows\\System32\\nvml.dll",
}

var TegraLinuxGlobs = []string{
	"/usr/local/cuda/lib64/libcudart.so*",
	"/usr/local/cuda/lib*/libnvidia-ml.so*",
	"/usr/local/cuda*/targets/aarch64-linux/lib/libcudart.so*",
}

var RocmLinuxGlobs = []string{
	"/opt/rocm*/lib*/librocm_smi64.so*",
}

var RocmWindowsGlobs = []string{
	"c:\\Windows\\System32\\rocm_smi64.dll",
}

// Jetson devices have JETSON_JETPACK="x.y.z" factory set to the Jetpack version installed.

var CudaTegra string = os.Getenv("JETSON_JETPACK")

// Note: gpuMutex must already be held
func initGPUHandles() {

	// TODO - if the ollama build is CPU only, don't do these checks as they're irrelevant and confusing

	gpuHandles = &handles{nil, nil, nil}
	var cudaMgmtName string
	var cudaMgmtPatterns []string
	var rocmMgmtName string
	var rocmMgmtPatterns []string	
	var tegraMgmtName string
	var tegraMgmtPatterns []string
	switch runtime.GOOS {
	case "windows":
		cudaMgmtName = "nvml.dll"
		cudaMgmtPatterns = make([]string, len(CudaWindowsGlobs))
		copy(cudaMgmtPatterns, CudaWindowsGlobs)
		rocmMgmtName = "rocm_smi64.dll"
		rocmMgmtPatterns = make([]string, len(RocmWindowsGlobs))
		copy(rocmMgmtPatterns, RocmWindowsGlobs)
	case "linux":
		cudaMgmtName = "libnvidia-ml.so"
		cudaMgmtPatterns = make([]string, len(CudaLinuxGlobs))
		copy(cudaMgmtPatterns, CudaLinuxGlobs)
		tegraMgmtName = "libcudart.so"
		tegraMgmtPatterns = make([]string, len(TegraLinuxGlobs))
		copy(tegraMgmtPatterns, TegraLinuxGlobs)
		rocmMgmtName = "librocm_smi64.so"
		rocmMgmtPatterns = make([]string, len(RocmLinuxGlobs))
		copy(rocmMgmtPatterns, RocmLinuxGlobs)
	default:
		return
	}

	slog.Info("Detecting GPU type")
	cudaLibPaths := FindGPULibs(cudaMgmtName, cudaMgmtPatterns)
	if len(cudaLibPaths) > 0 {
		cuda := LoadCUDAMgmt(cudaLibPaths)
		if cuda != nil {
			slog.Info("Nvidia GPU detected")
			gpuHandles.cuda = cuda
			return
		}
	}

	tegraLibPaths := FindGPULibs(tegraMgmtName, tegraMgmtPatterns)
	if CudaTegra != "" && len(tegraLibPaths) > 0 {
		tegra := LoadTEGRAMgmt(tegraLibPaths)
		if tegra != nil {
			slog.Info("Nvidia Tegra SOC detected")
			gpuHandles.tegra = tegra
			return
		}
	}

	rocmLibPaths := FindGPULibs(rocmMgmtName, rocmMgmtPatterns)
	if len(rocmLibPaths) > 0 {
		rocm := LoadROCMMgmt(rocmLibPaths)
		if rocm != nil {
			slog.Info("Radeon GPU detected")
			gpuHandles.rocm = rocm
			return
		}
	}
}

func GetGPUInfo() GpuInfo {
	// TODO - consider exploring lspci (and equivalent on windows) to check for
	// GPUs so we can report warnings if we see Nvidia/AMD but fail to load the libraries
	gpuMutex.Lock()
	defer gpuMutex.Unlock()
	if gpuHandles == nil {
		initGPUHandles()
	}

	// All our GPU builds on x86 have AVX enabled, so fallback to CPU if we don't detect at least AVX
	cpuVariant := GetCPUVariant()
	if cpuVariant == "" && runtime.GOARCH == "amd64" {
		slog.Warn("CPU does not have AVX or AVX2, disabling GPU support.")
	}

	var memInfo C.mem_info_t
	resp := GpuInfo{}
	if gpuHandles.cuda != nil && (cpuVariant != "" || runtime.GOARCH != "amd64") {
		C.cuda_check_vram(*gpuHandles.cuda, &memInfo)
		if memInfo.err != nil {
			slog.Info(fmt.Sprintf("error looking up CUDA GPU memory: %s", C.GoString(memInfo.err)))
			C.free(unsafe.Pointer(memInfo.err))
		} else if memInfo.count > 0 {
			// Verify minimum compute capability
			var cc C.cuda_compute_capability_t
			C.cuda_compute_capability(*gpuHandles.cuda, &cc)
			if cc.err != nil {
				slog.Info(fmt.Sprintf("error looking up CUDA GPU compute capability: %s", C.GoString(cc.err)))
				C.free(unsafe.Pointer(cc.err))
			} else if cc.major > CudaComputeMin[0] || (cc.major == CudaComputeMin[0] && cc.minor >= CudaComputeMin[1]) {
				slog.Info(fmt.Sprintf("CUDA Compute Capability detected: %d.%d", cc.major, cc.minor))
				resp.Library = "cuda"
			} else {
				slog.Info(fmt.Sprintf("CUDA GPU is too old. Falling back to CPU mode. Compute Capability detected: %d.%d", cc.major, cc.minor))
			}
		}
	} else if gpuHandles.tegra != nil {
		// C.tegra_check_vram(*gpuHandles.tegra, &memInfo) // Tegra is iGPU, should gather data similar to MacOS
		mem, _ := getCPUMem()
		memInfo.free = C.uint64_t(mem.FreeMemory)
		memInfo.total = C.uint64_t(mem.TotalMemory)
		// Verify minimum compute capability
		var tcc C.tegra_compute_capability_t
		C.tegra_compute_capability(*gpuHandles.tegra, &tcc)
		if tcc.err != nil {
			slog.Info(fmt.Sprintf("error looking up Tegra SOC CUDA compute capability: %s", C.GoString(tcc.err)))
			C.free(unsafe.Pointer(tcc.err))
		} else if tcc.major > CudaComputeMin[0] || (tcc.major == CudaComputeMin[0] && tcc.minor >= CudaComputeMin[1]) {
			slog.Info(fmt.Sprintf("Tegra CUDA Compute Capability detected: %d.%d", tcc.major, tcc.minor))
			memInfo.count += 1
			resp.Library = "tegra"
		} else {
			slog.Info(fmt.Sprintf("Tegra CUDA GPU is too old. Falling back to CPU mode. Compute Capability detected: %d.%d", tcc.major, tcc.minor))
		}
	} else if gpuHandles.rocm != nil && (cpuVariant != "" || runtime.GOARCH != "amd64") {
		C.rocm_check_vram(*gpuHandles.rocm, &memInfo)
		if memInfo.err != nil {
			slog.Info(fmt.Sprintf("error looking up ROCm GPU memory: %s", C.GoString(memInfo.err)))
			C.free(unsafe.Pointer(memInfo.err))
		} else if memInfo.igpu_index >= 0 && memInfo.count == 1 {
			// Only one GPU detected and it appears to be an integrated GPU - skip it
			slog.Info("ROCm unsupported integrated GPU detected")
		} else if memInfo.count > 0 {
			if memInfo.igpu_index >= 0 {
				// We have multiple GPUs reported, and one of them is an integrated GPU
				// so we have to set the env var to bypass it
				// If the user has specified their own ROCR_VISIBLE_DEVICES, don't clobber it
				val := os.Getenv("ROCR_VISIBLE_DEVICES")
				if val == "" {
					devices := []string{}
					for i := 0; i < int(memInfo.count); i++ {
						if i == int(memInfo.igpu_index) {
							continue
						}
						devices = append(devices, strconv.Itoa(i))
					}
					val = strings.Join(devices, ",")
					os.Setenv("ROCR_VISIBLE_DEVICES", val)
				}
				slog.Info(fmt.Sprintf("ROCm integrated GPU detected - ROCR_VISIBLE_DEVICES=%s", val))
			}
			resp.Library = "rocm"
			var version C.rocm_version_resp_t
			C.rocm_get_version(*gpuHandles.rocm, &version)
			verString := C.GoString(version.str)
			if version.status == 0 {
				resp.Variant = "v" + verString
			} else {
				slog.Info(fmt.Sprintf("failed to look up ROCm version: %s", verString))
			}
			C.free(unsafe.Pointer(version.str))
		}
	}
	if resp.Library == "" {
		C.cpu_check_ram(&memInfo)
		resp.Library = "cpu"
		resp.Variant = cpuVariant
	}
	if memInfo.err != nil {
		slog.Info(fmt.Sprintf("error looking up CPU memory: %s", C.GoString(memInfo.err)))
		C.free(unsafe.Pointer(memInfo.err))
		return resp
	}

	resp.DeviceCount = uint32(memInfo.count)
	resp.FreeMemory = uint64(memInfo.free)
	resp.TotalMemory = uint64(memInfo.total)
	return resp
}

func getCPUMem() (memInfo, error) {
	var ret memInfo
	var info C.mem_info_t
	C.cpu_check_ram(&info)
	if info.err != nil {
		defer C.free(unsafe.Pointer(info.err))
		return ret, fmt.Errorf(C.GoString(info.err))
	}
	ret.FreeMemory = uint64(info.free)
	ret.TotalMemory = uint64(info.total)
	return ret, nil
}

func CheckVRAM() (int64, error) {
	gpuInfo := GetGPUInfo()
	if gpuInfo.FreeMemory > 0 && (gpuInfo.Library == "cuda" || gpuInfo.Library == "rocm") {
		// leave 10% or 1024MiB of VRAM free per GPU to handle unaccounted for overhead
		overhead := gpuInfo.FreeMemory / 10
		gpus := uint64(gpuInfo.DeviceCount)
		if overhead < gpus*1024*1024*1024 {
			overhead = gpus * 1024 * 1024 * 1024
		}
		avail := int64(gpuInfo.FreeMemory - overhead)
		slog.Debug(fmt.Sprintf("%s detected %d devices with %dM available memory", gpuInfo.Library, gpuInfo.DeviceCount, avail/1024/1024))
		return avail, nil
	} else if gpuInfo.FreeMemory > 0 && gpuInfo.Library == "tegra" {
		// Tegra SOCs are iGPUs and share GPU memory with system memory.
		systemMemory := int64(memory.TotalMemory())

		// TODO: handle case where iogpu.wired_limit_mb is set to a higher value
		if systemMemory <= 36*1024*1024*1024 {
			systemMemory = systemMemory * 2 / 3
		} else {
			systemMemory = systemMemory * 3 / 4
		}

		return systemMemory, nil
	}

	return 0, fmt.Errorf("no GPU detected") // TODO - better handling of CPU based memory determiniation
}

func FindGPULibs(baseLibName string, patterns []string) []string {
	// Multiple GPU libraries may exist, and some may not work, so keep trying until we exhaust them
	var ldPaths []string
	gpuLibPaths := []string{}
	slog.Info(fmt.Sprintf("Searching for GPU management library %s", baseLibName))

	switch runtime.GOOS {
	case "windows":
		ldPaths = strings.Split(os.Getenv("PATH"), ";")
	case "linux":
		ldPaths = strings.Split(os.Getenv("LD_LIBRARY_PATH"), ":")
	default:
		return gpuLibPaths
	}
	// Start with whatever we find in the PATH/LD_LIBRARY_PATH
	for _, ldPath := range ldPaths {
		d, err := filepath.Abs(ldPath)
		if err != nil {
			continue
		}
		patterns = append(patterns, filepath.Join(d, baseLibName+"*"))
	}
	slog.Debug(fmt.Sprintf("gpu management search paths: %v", patterns))
	for _, pattern := range patterns {
		// Ignore glob discovery errors
		matches, _ := filepath.Glob(pattern)
		for _, match := range matches {
			// Resolve any links so we don't try the same lib multiple times
			// and weed out any dups across globs
			libPath := match
			tmp := match
			var err error
			for ; err == nil; tmp, err = os.Readlink(libPath) {
				if !filepath.IsAbs(tmp) {
					tmp = filepath.Join(filepath.Dir(libPath), tmp)
				}
				libPath = tmp
			}
			new := true
			for _, cmp := range gpuLibPaths {
				if cmp == libPath {
					new = false
					break
				}
			}
			if new {
				gpuLibPaths = append(gpuLibPaths, libPath)
			}
		}
	}
	slog.Info(fmt.Sprintf("Discovered GPU libraries: %v", gpuLibPaths))
	return gpuLibPaths
}

func LoadCUDAMgmt(cudaLibPaths []string) *C.cuda_handle_t {
	var resp C.cuda_init_resp_t
	resp.ch.verbose = getVerboseState()
	for _, libPath := range cudaLibPaths {
		lib := C.CString(libPath)
		defer C.free(unsafe.Pointer(lib))
		C.cuda_init(lib, &resp)
		if resp.err != nil {
			slog.Info(fmt.Sprintf("Unable to load CUDA management library %s: %s", libPath, C.GoString(resp.err)))
			C.free(unsafe.Pointer(resp.err))
		} else {
			return &resp.ch
		}
	}
	return nil
}

func LoadTEGRAMgmt(tegraLibPaths []string) *C.tegra_handle_t {
	var resp C.tegra_init_resp_t
	resp.th.verbose = getVerboseState()
	for _, libPath := range tegraLibPaths {
		lib := C.CString(libPath)
		defer C.free(unsafe.Pointer(lib))
		C.tegra_init(lib, &resp)
		if resp.err != nil {
			slog.Info(fmt.Sprintf("Unable to load Tegra CUDA management library %s: %s", libPath, C.GoString(resp.err)))
			C.free(unsafe.Pointer(resp.err))
		} else {
			return &resp.th
		}
	}
	return nil
}

func LoadROCMMgmt(rocmLibPaths []string) *C.rocm_handle_t {
	var resp C.rocm_init_resp_t
	resp.rh.verbose = getVerboseState()
	for _, libPath := range rocmLibPaths {
		lib := C.CString(libPath)
		defer C.free(unsafe.Pointer(lib))
		C.rocm_init(lib, &resp)
		if resp.err != nil {
			slog.Info(fmt.Sprintf("Unable to load ROCm management library %s: %s", libPath, C.GoString(resp.err)))
			C.free(unsafe.Pointer(resp.err))
		} else {
			return &resp.rh
		}
	}
	return nil
}

func getVerboseState() C.uint16_t {
	if debug := os.Getenv("OLLAMA_DEBUG"); debug != "" {
		return C.uint16_t(1)
	}
	return C.uint16_t(0)
}
