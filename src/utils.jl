@inline function compute_shmem_size(d, Bs, T)
    return (d * Bs * 3 + Bs * 3 + Bs * Bs * 2) * sizeof(T)
end
