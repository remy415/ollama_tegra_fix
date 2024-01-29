# Ollama Tegra Fix

Things you need to do:

1. git clone ollama, including the recursive flag to get llama_cpp


2. Ensure all files in package_cudart_build are copied into the ollama base directory / overwriting their files.

Also ensure: ***IMPORTANT***
```-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc```
is in the llm/generate/gen_linux.sh file under CUBLAS; it didn't compile until I hard-coded that value into the gen_linux script.

```
cd package_cudart_build
cp -r ./* ../ollama/
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