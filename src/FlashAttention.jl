module FlashAttention

using CUDA

include("utils.jl")
include("flash_attention.jl")

export flash_attention, setMaxShmem

end
