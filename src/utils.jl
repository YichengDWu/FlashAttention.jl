@inline function compute_shmem_size(d, Bs, T)
    return ((d + 2) * Bs * 3 + Bs * Bs) * sizeof(T)
end

"""
    setMaxShmem(shmem)

Set the maximum shared memory size for the current device to `shmem` KB.
"""
function setMaxShmem(shmem)
    kernel = cufunction(flash_attention_kernel, NTuple{4, CuDeviceArray{Float16, 4, 1}})
    return CUDA.cuFuncSetAttribute(kernel.fun,
                                   CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                   shmem * 1024)
end
