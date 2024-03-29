# FlashAttention

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://YichengDWu.github.io/FlashAttention.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://YichengDWu.github.io/FlashAttention.jl/dev/)
[![Build Status](https://github.com/YichengDWu/FlashAttention.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/YichengDWu/FlashAttention.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/YichengDWu/FlashAttention.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/YichengDWu/FlashAttention.jl)

This is a Julia implementation of the [Flash Attention algorithm](https://arxiv.org/pdf/2205.14135v2.pdf).

## Usage
```julia
using FlashAttention, CUDA

Q = CUDA.randn(Float16, 64, 1024, 48, 3);
K = CUDA.randn(Float16, 64, 1024, 48, 3);
V = CUDA.randn(Float16, 64, 1024, 48, 3);

flash_attention(Q,K,V)
```
## Profiling

Please refer to the file `flash_attention.ncu-rep`. This is not the fastest implementation for 
1) we do **not** use tensor cores as in the C++ implmentation,
2) CUDA.jl doese not yet support asynchronous copy from global memory to shared memory, and
3) this kernel's theoretical occupancy (12.5%) is limited by the required amount of shared memory.

## Future work
I plan to implement it in the future using [MoYe.jl](https://github.com/YichengDWu/MoYe.jl) to achieve competitive performance.
