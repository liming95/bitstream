#include <stdio.h>
#include <cuda_runtime.h>

#define SET_HIGH_BIT(number, length) (number | (1 << (length-1)))
#define CLEAR_LOW_BIT(number) (number ^ 1)
#define grid_size 2
#define block_size 2
#define thread_size 2

/* function: bit_stream << shift_count
 * size: the size of uint32_t
 * shift_count: shift number
 * shift_global: shift in and shift out bit
 */
__global__ void shift_left(uint32_t *bit_stream, int size, int shift_count, uint32_t* shift_global){
    __shared__ shift_block[thread_size];
    //block inner
    int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t stream = bit_stream[thread_idx];
    if(threadIdx.x == 0 & blockIdx.x == 0){
        shift_block[0] |= shift_global[0];
    }

    if(threadIdx.x < blockDim.x-1){
        shift_block[threadIdx.x+1] = bit_stream[thread_idx] >> (size - shift_count);
    }
    else{
        shift_global[(blockIdx.x+1)%gridDim.x] |= bit_stream[thread_idx] >> (size - shift_count);
    }

    if(threadIdx.x == 0 & blockIdx.x != 0){
        shift_block[0] = shift_blobal[blockIdx.x];
    }

    bit_stream[thread_idx];
    //block inter
}

void get_bit_stream(uint32_t *bit_stream, int size){
    for(int i = 0; i < size; i++){
        bit_stream[i] = SET_HIGH_BIT(bit_steam[i], size);
    }
}

int main()
{
    int shift_count = 1;
    int iteration_time = grid_size;
    int grid_dim = block_size * thread_size;
    int bit_stream_size = grid_size * block_size * thread_size;


    uint32_t bit_stream[bit_stream_size] = {0, 1, 2, 4, 8, 16, 32, 64};
    uint32_t bit_stream_shift[bit_stream_size] = {0, 2, 4, 8, 16, 32, 64, 128};
    uint32_t *bit_stream_gpu;
    uint32_t shift_global[block_size] = {0};
    uint32_t *shift_global_gpu;

    cudaMalloc((void **)&bit_stream_gpu, sizeof(uint32_t)*grid_dim);
    cudaMalloc((void **)&shift_global_gpu, sizeof(uint32_t)*block_size);
    *shift_global = 0;

    cudaMemcpy(shift_global_gpu, shift_global, sizeof(uint32_t, cudaMemcpyHostToDevice));
    get_bit_stream(bit_stream, bit_stream_size);



    for(int i = 0; i < iteration_time; i++){
        cudaMemcpy(bit_stream_gpu, bit_stream+i*grid_dim,
            grid_dim*sizeof(uint32_t), cudaMemcpyHostToDevice);

        shift_left<<<block_size, thread_size>>>(bit_stream_gpu, sizeof(uint32_t), shift_count, shift_global_gpu);

        cudaMemcpy(bit_stream+i*grid_dim, bit_stream_gpu,
            grid_dim*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }

    for(int i = 0; i < bit_stream_size; i++){
        if(CLEAR_LOW_BIT(bit_stream[i]) == bit_stream_shift[i]){
            printf("[%d]: {bit_stream:%u, shift_global_stream:%u", i,
                bit_stream[i] >> shift_count, bit_stream_shift[i]);
        }
        else{
            printf("ERROR[%d]: {bit_stream:%u, shift_global_stream:%u}", i,
                bit_stream[i] >> shift_count, bit_stream_shift[i]);
        }
    }

    cudaFree(shift_global_gpu);
    cudaFree(bit_stream_gpu);

    return 0;

}
