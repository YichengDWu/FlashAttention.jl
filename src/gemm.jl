function gemm_kernel(A,B,C)
    d = size(A,2)
    dim = blockDim().x
    SA = CUDA.CuDynamicSharedArray(eltype(A), (dim, dim)) # 共享内存
    SB = CUDA.CuDynamicSharedArray(eltype(B), (dim, dim), sizeof(SA))

    row = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    col = (blockIdx().y - 1)*blockDim().y + threadIdx().y

    tx = threadIdx().x
    ty = threadIdx().y

    tmp = zero(eltype(A))
    for i in 1:cld(d, dim)
        SA[tx, ty] = A[row, ty+(i-1)*dim]
        SB[tx, ty] = B[tx+(i-1)*dim, col]
        sync_threads()
        for j in 1:dim
            tmp += SA[tx, j] * SB[j, ty]
        end
        sync_threads()
    end

    C[row, col] = tmp

    return nothing
end

function gemm_kernel2(A,B,C)
    d = size(A,2)
    dim_x = blockDim().x
    dim_y = blockDim().y
    SA = CUDA.CuDynamicSharedArray(eltype(A), (dim_x, d)) # 共享内存
    SB = CUDA.CuDynamicSharedArray(eltype(B), (d, dim_y), sizeof(SA))

    row = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    col = (blockIdx().y - 1)*blockDim().y + threadIdx().y

    tx = threadIdx().x
    ty = threadIdx().y

    for i in 1:cld(d, dim_x)
        SA[tx, ty+(i-1)*dim_x] = A[row, ty+(i-1)*dim_x]
    end

    for i in 1:cld(d, dim_y)
        SB[tx+(i-1)*dim_y, ty] = B[tx+(i-1)*dim_y, col]
    end
    sync_threads()

    tmp = zero(eltype(A))
    for j in 1:d
        tmp += SA[tx, j] * SB[j, ty]
    end
    C[row, col] = tmp

    return nothing
end

N = 1024
d = 64
A = CUDA.rand(N,d)
B = CUDA.rand(d,N)
C = CUDA.zeros(N,N)

Block = 32
Ta = cld(N, Block)
Tb = cld(N, Block)

shmem = Block * d * sizeof(eltype(A)) * 2
@cuda threads=(Block, Block) blocks=(Ta,Tb) shmem = shmem gemm_kernel2(A,B,C)
