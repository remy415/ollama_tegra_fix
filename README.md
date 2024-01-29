# Ollama Tegra Fix

Things you need to do:

1. git clone 

Ensure all files in package_cudart_build are copied into the ollama base directory / overwriting their files.

```
cd package_cudart_build
cp -r ./* ../ollama/
```

Set some necessary ENV variables:
```
export OLLAMA_LLM_LIBRARY='cuda_v11'
export OLLAMA_SKIP_CPU_GENERATE='yes'
```

Ensure
-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
is in the llm/generate/gen_linux.sh file under CUBLAS; it didn't compile until I hard-coded that value into the gen_linux script.
