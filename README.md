# Ollama Tegra Fix
## Important ENV Vars for building. Note: I still need to validate which are required, this is WIP:
```
# On my Jetson Orin Nano, Jetpack 5 comes with CUDA 11-4 with 11-8 compatibility installed. The LD_LIBRARY_PATH
# was required to tell the compiler where the "compat" libcuda.so is kept, otherwise it defaulted to the
# CUDA 11-4 libcuda.so and crashed on 'ollama run'.
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/compat:/usr/local/cuda/include"
export CUDA_LIB_DIR="/usr/local/cuda/lib64"
export CGO_FLAGS="-g"
export CMAKE_CUDA_COMPILER="/usr/local/cuda/bin/nvcc"
export CUDACXX="/usr/local/cuda/bin/nvcc"
export OLLAMA_LLM_LIBRARY="cuda_v11"
export OLLAMA_SKIP_CPU_GENERATE="yes"
```

*** NOTE *** This part of the guide is deprecated and likely doesn't work. I'm working on updating it, see above.

Compiling llama_cpp with CUDA enabled requires CMAKE 3.17 and higher. On Jetpack 5, it will require manual
installation of cmake.

Things you need to do:

1. git clone ollama including the recursive flag to get llama_cpp. Clone this repo.
```
git clone --depth=1 --recursive https://github.com/ollama/ollama.git
git clone https://github.com/remy415/ollama_tegra_fix.git
```

2. Ensure all files in package_cudart_build are copied into the ollama base directory / overwriting their files.

Also ensure: ***IMPORTANT***
```-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc```
is in the llm/generate/gen_linux.sh file under CUBLAS; it didn't compile until I hard-coded that value into the gen_linux script.

```
cd ollama_tegra_fix/package_cudart_build
cp -r ./* ../../ollama/
```

3. Set some necessary ENV variables:

```
export OLLAMA_LLM_LIBRARY='cuda_v11'
export OLLAMA_SKIP_CPU_GENERATE='yes'
```

4. From the ollama base directory:

```
go generate ./...
go build .
```

5. Follow bnodnarb's guide here:

https://github.com/ollama/ollama/blob/main/docs/tutorials/nvidia-jetson.md

Especially crucial is the creation of the mistral-jetson model as it forces GPU to load.
