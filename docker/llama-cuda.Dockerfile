# llama.cpp's server built for an older NVIDIA driver (the official image needs CUDA 12.8 / driver 570+).
# Used by `bash scripts/train_local.sh basetest-gguf …` when the driver is older; built once (~10-20 min):
#   docker build -f docker/llama-cuda.Dockerfile --build-arg CUDA=12.2.2 -t virgo-llama:cuda12.2 docker
ARG CUDA=12.2.2
FROM nvidia/cuda:${CUDA}-devel-ubuntu22.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential libcurl4-openssl-dev libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ARG LLAMA_REF=master
RUN git clone --depth 1 --branch ${LLAMA_REF} https://github.com/ggml-org/llama.cpp /src
WORKDIR /src
# 89 = RTX 40xx (Ada); add more with --build-arg ARCHS="86;89" for other cards.
ARG ARCHS=89
RUN cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${ARCHS}" -DLLAMA_CURL=ON -DLLAMA_BUILD_TESTS=OFF \
      -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build --config Release -j 8 --target llama-server

FROM nvidia/cuda:${CUDA}-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends libcurl4 libgomp1 curl ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/bin/ /app/
ENV LD_LIBRARY_PATH=/app
ENTRYPOINT ["/app/llama-server"]
