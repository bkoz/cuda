# CUDA Sandbox

A mini-workshop to learn about compiling and running simple CUDA programs on RHEL.

- Order the Base Red Hat AI Inference Server (RHAIIS) from the demo catalog.
- Perform the following to prepare the system to compile and run CUDA programs.
  - `export PATH=$PATH:/usr/local/cuda/bin`
  - Even better, modify your `~/.bashrc`

- Clone https://github.com/harrism/nsys_easy
	- Move the `nsys_easy` script into a directory contained in $PATH
	- `mkdir $HOME/.local/bin` is a good option
- Compiling and running programs

`nvcc add_cuda.cu -o add_cuda`

`add_cuda`

- Run the profiler.

```bash
nsys_easy <executable>
```

Containers core dump cuda programs. I need to investigate CUDA revisions
between the container and host.

```bash
podman run -it --rm -v $(pwd):/scratch:z nvcr.io/nvidia/cuda-dl-base:25.06-cuda12.9-devel-ubuntu24.04 bash
```

#### References
https://developer.nvidia.com/blog/even-easier-introduction-cuda/

https://github.com/harrism/mini-nbody.git
