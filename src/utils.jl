@inline function compute_shmem_size(d, Bs, T)
    return (Bs * d * 3 + 4 * d + Bs * Bs) * sizeof(T)
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

function _checkbounds(Q, K, V)
    sQ, sK, sV = size(Q), size(K), size(V)
    sK != sV && throw(DimensionMismatch("K and V must have the same shape"))
    sQ[3:4] != sK[3:4] != sV[3:4] &&
        throw(DimensionMismatch("Q, K and V must have the same batch size and head size"))
    return sQ[1] != sK[2] != sV[2] &&
           throw(DimensionMismatch("Q, K and V must have the same hidden dimension"))
end


@inline function mod1_pow2(x, y)
    r = x & (y - 1)
    return ifelse(r == 0, y, r)
end
