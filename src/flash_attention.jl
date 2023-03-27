EPSILON = 1e-10
MASK_VALUE = -1e10

Q_CHUNK_SIZE = 1024
K_CHUNK_SIZE = 1024

function _flash_attention(q,k,v,mask)
    dim, q_len, heads, batch = size(q)
    v_dim = size(v, 1)

    function chunk_scanner(chunk_idx)
        chunk_size = min(Q_CHUNK_SIZE, q_len)
        q_chunk = @view q[:, :, :, chunk_idx:chunk_idx+chunk_size-1]
        return (chunk_idx + chunk_size, _query_chunk_flash_attention(chunk_idx, q_chunk, k, v, key_mask))


end
