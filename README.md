# Ollama Tegra Fix
## SEE `https://github.com/remy415/ollama` FOR AUTOMATED BUILD. 

### ENSURE THESE TWO VARIABLES ARE SET:
`export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/compat:/usr/local/cuda/include"`
`export OLLAMA_SKIP_CPU_GENERATE="1"`

## Also, Tegra devices will fail `go generate ./...` unless architectures are set:
### L4T_VERSION.major >= 36:    # JetPack 6
`export CUDA_ARCHITECTURES="87"`
### L4T_VERSION.major >= 34:  # JetPack 5
`export CUDA_ARCHITECTURES="72;87"`
### L4T_VERSION.major == 32:  # JetPack 4
`export CUDA_ARCHITECTURES="53;62;72"`
