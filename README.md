# Ollama Tegra Fix (DEPRECATED)
## SEE `https://github.com/remy415/ollama` FOR AUTOMATED BUILD. ENSURE THESE TWO VARIABLES ARE SET:
#### `export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/compat:/usr/local/cuda/include"`
#### `export OLLAMA_SKIP_CPU_GENERATE="1"`

# Tegra devices will fail generate unless architectures are set:
        # Nano/TX1 = 5.3, TX2 = 6.2, Xavier = 7.2, Orin = 8.7
        # L4T_VERSION.major >= 36:    # JetPack 6
        #     CUDA_ARCHITECTURES="87"
        # L4T_VERSION.major >= 34:  # JetPack 5
        #     CUDA_ARCHITECTURES="72;87"
        # L4T_VERSION.major == 32:  # JetPack 4
        #     CUDA_ARCHITECTURES="53;62;72"
