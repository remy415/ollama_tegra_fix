ARG GOLANG_VERSION=1.21.3
ARG CMAKE_VERSION=3.22.1
ARG L4T_VERSION=r35.4.1
ARG OLLAMA_REPOSITORY="ollama"
ARG OLLAMA_BRANCH="main"

# Copy the minimal context we need to run the generate scripts
FROM dustynv/llama_cpp:gguf-${L4T_VERSION} AS ollama-cuda-l4t-base
ARG CMAKE_VERSION
ARG GOLANG_VERSION
ARG OLLAMA_REPOSITORY
WORKDIR /go/src/github.com/jmorganca/
COPY ./scripts/tegra_linux_deps.sh /opt/
RUN CMAKE_VERSION=${CMAKE_VERSION} GOLANG_VERSION=${GOLANG_VERSION} sh /opt/tegra_linux_deps.sh
ENV PATH /usr/local/bin:/usr/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64/:/usr/local/cuda/include
ENV OLLAMA_SKIP_CPU_GENERATE="yes"
ENV CGO_ENABLED="1"
ENV JETSON_JETPACK="5.1.2"
ENV CMAKE_CUDA_ARCHITECTURES="72;87"
ADD https://api.github.com/repos/${OLLAMA_REPOSITORY}/ollama/git/refs/heads/${OLLAMA_BRANCH} version.json
RUN git clone --depth=1 --recursive https://github.com/${OLLAMA_REPOSITORY}/ollama.git
WORKDIR /go/src/github.com/jmorganca/ollama
RUN go generate ./...
RUN go build .

# Runtime stages
FROM dustynv/llama_cpp:gguf-${L4T_VERSION} as runtime-arm64
COPY ./scripts/tegra_linux_deps.sh /opt/
RUN apt-get update && apt-get install -y ca-certificates && CMAKE_VERSION=${CMAKE_VERSION} GOLANG_VERSION=${GOLANG_VERSION} sh /opt/tegra_linux_deps.sh
COPY --from=ollama-cuda-l4t-base /go/src/github.com/jmorganca/ollama/ollama /bin/ollama
EXPOSE 11434
ENV OLLAMA_HOST 0.0.0.0
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/cuda/include
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENTRYPOINT ["/bin/ollama"]
CMD ["serve"]
