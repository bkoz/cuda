# CUDA Sandbox

A mini-workshop to learn about compiling and running simple CUDA programs on RHEL.

- Order the Base Red Hat AI Inference Server (RHAIIS) from the demo catalog.
- Perform the following to prepare the system to compile and run CUDA programs.
  - `export PATH=$PATH:/usr/local/cuda/bin`
  - Even better, modify your `~/.bashrc`

- git clone https://github.com/harrism/nsys_easy
- cd `nsys_easy`
- `mkdir $HOME/.local/bin` is a good option
- `cp nsys_easy $HOME/.local/bin`

- Compiling and running programs

`nvcc add_cuda.cu -o add_cuda`

`./add_cuda`

- Run the profiler.

```bash
nsys_easy add_cuda
```

Containers core dump cuda programs. I need to investigate CUDA revisions
between the container and host.

```bash
podman run -it --rm -v $(pwd):/scratch:z nvcr.io/nvidia/cuda-dl-base:25.06-cuda12.9-devel-ubuntu24.04 bash
```

#### References
https://developer.nvidia.com/blog/even-easier-introduction-cuda/

https://github.com/harrism/mini-nbody.git
