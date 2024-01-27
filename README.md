# Ollama Tegra Fix
Quick draft for an idea to compile custom shared-object library to query CUDA API on Tegra devices (Jetson). gpu.go file was modified to include new libtegra-ml.so binary, an os.getenv query to check if JETSON_JETPACK env variable is set (variable set by default by system), and report Tegra GPU detection.

shared-object compiled on a Jetson Orin Nano 8g, running Jetpack 5.1.2 / L4T 35.4.1 / CUDA 11.8 with the nvcc compiler.

This is a work in progress.
