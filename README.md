# Ollama Tegra Fix (DEPRECATED)
## SEE `https://github.com/remy415/ollama` FOR AUTOMATED BUILD. ENSURE THESE TWO VARIABLES ARE SET
## AND DONT SET THE REST:
## `export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/compat:/usr/local/cuda/include"`
## `export OLLAMA_SKIP_CPU_GENERATE="1"`
## Required environment setup
On Jetpack 5, you will need to manually install the new version of cmake. 
You can try using pip to install it, YMMV.

```
cmake >= 3.17 # Compiling llama_cpp with CUBLAS/CUDA enabled requires CMAKE 3.17 or higher. 
gcc >= 9 # CUDA 11.8 CPP requires gcc/g++ >= 9
golang >= v1.26.6 # This is just what I used, I don't know the minimum Golang requirement.
```

## Important ENV Vars for building. Note: this is a WIP, I am still validating ENV vars.

```
# On my Jetson Orin Nano, Jetpack 5 comes with CUDA 11-4 with
# 11-8 compatibility installed. The LD_LIBRARY_PATH setting
# was required to tell the compiler where the "compat" libcuda.so
# is kept, otherwise it defaulted to the CUDA 11-4 libcuda.so and
# crashed on 'ollama run'.

# IMPORTANT: THE COMPILER NEEDS ALL 3 LD_LIBRARY_PATH PATHS.
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/compat:/usr/local/cuda/include" 
export CGO_FLAGS="-g"
export OLLAMA_LLM_LIBRARY="cuda_v11" 
export OLLAMA_SKIP_CPU_GENERATE="1" # Might as well set this since ARM SOCs don't support AVX
```

*** NOTE *** Below this part of the guide is deprecated and likely doesn't work. I'm working on updating it, see above.

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
