//go:build tegra

package gpu

import (
	"fmt"
	"log/slog"

	"gorgonia.org/cu"
)

type memInfo struct {
	TotalMemory uint64 `json:"total_memory,omitempty"`
	FreeMemory  uint64 `json:"free_memory,omitempty"`
	DeviceCount uint32 `json:"device_count,omitempty"`
}

// Beginning of an `ollama info` command
type GpuInfo struct {
	memInfo
	Library string `json:"library,omitempty"`

	// Optional variant to select (e.g. versions, cpu feature flags)
	Variant string `json:"variant,omitempty"`

	// TODO add other useful attributes about the card here for discovery information
}

var CudaComputeMin = [2]int{5, 0}

func GetGPUInfo() GpuInfo {
	memInfo := memInfo{}
	resp := GpuInfo{}
	count, err := cu.NumDevices()
	if err != nil {
		slog.Info("error querying number of devices")
		return resp
	}

	memInfo.DeviceCount = count

	// Tegra devices have only one GPU at index 0
	d := 0

	totalMem, tmerr := cu.Device(d).TotalMem()
	freeMem, fmerr := cu.Device(d).FreeMemory()

	if tmerr != nil || fmerr != nil {
		slog.Info("error looking up CUDA GPU memory")
	} else {
		memInfo.TotalMemory = totalMem
		memInfo.FreeMemory = freeMem
		ccmaj, ccmajerr := cu.Device(d).Attribute(cu.ComputeCapabilityMajor)
		ccmin, ccminerr := cu.Device(d).Attribute(cu.ComputeCapabilityMinor)
		if ccmajerr != nil || ccminerr != nil {
			slog.Info("error looking up CUDA GPU compute capability")
		} else if ccmaj > CudaComputeMin[0] || (ccmaj == CudaComputeMin[0] && ccmin >= CudaComputeMin[1]) {
			slog.Info(fmt.Sprintf("CUDA Compute Capability detected: %d.%d", ccmaj, ccmin))
			resp.Library = "cuda"
		} else {
			slog.Info(fmt.Sprintf("CUDA GPU is too old. Falling back to CPU mode. Compute Capability detected: %d.%d", ccmaj, ccmin))
		}
	}

	if resp.Library == "" {
		resp.Library = "cpu"
		resp.Variant = ""
	}

	return resp
}

// CheckVRAM returns the free VRAM in bytes on Linux machines with NVIDIA GPUs
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
	}

	return 0, fmt.Errorf("no GPU detected") // TODO - better handling of CPU based memory determiniation
}

func getCPUMem() (memInfo, error) {
	return memInfo{
		TotalMemory: 0,
		FreeMemory:  0,
		DeviceCount: 0,
	}, nil
}
