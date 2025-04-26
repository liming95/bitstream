/* compile: nvcc --cudart shared -o shift ../../bitop/shift.cu
 * time: 2025/04/02
 * author: lim(951238600@qq.com)
*/
#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define SET_HIGH_BIT(number, size) (number | (unsigned int)(1 << (size*8-1)))
#define CLEAR_HIGH_BIT(number, size) (number & (~(unsigned int)(1 << (size*8-1))))
#define CLEAR_LOW_BIT(number) (number & (~1))
//#define WAIT_GLOBAL(number, size, condition) while((__ldg(&number) >> (size*8-1)) != condition)
#define WAIT(number, size, condition) while(((volatile uint32_t&)number >> (size*8-1)) != condition) {}
#define GRID_NUM 2
#define BLOCK_NUMBER 2
#define THREAD_NUM_PER_BLOCK 2

/* function: bit_stream << shift_count
 * size: the size of uint32_t
 * shift_count: shift number
 * shift_global: shift in and shift out bit
 */
__global__ void shift_left(uint32_t *bit_stream, int size, int shift_count, uint32_t* shift_global){
    __shared__ uint32_t shift_block[THREAD_NUM_PER_BLOCK];
    int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int bit_length = size * 8;

    // Todo: multi parameter will print error data.
    // printf("thread[%u,%u,(%u, %u, %u)]: bit_stream[%d] = %u\n", blockIdx.x, threadIdx.x,
    //     blockDim.x, blockDim.y, blockDim.z, thread_idx, bit_stream[thread_idx]);
    if(threadIdx.x == 0 && blockIdx.x == 0){
        //device inter
        shift_block[0] |= shift_global[0];
        __threadfence();
        shift_block[0] = SET_HIGH_BIT(shift_block[0], sizeof(uint32_t));
        shift_global[0] = CLEAR_HIGH_BIT(shift_global[0], sizeof(uint32_t));

    }

    if(threadIdx.x < blockDim.x-1){
        //block inner
        shift_block[threadIdx.x+1] = bit_stream[thread_idx] >> (bit_length - shift_count);
        __threadfence_block();
        shift_block[threadIdx.x+1] = SET_HIGH_BIT(shift_block[threadIdx.x+1], sizeof(uint32_t));
    }
    else{
        //block inter & device inter
        int block_idx = (blockIdx.x + 1) % gridDim.x;
        WAIT(shift_global[block_idx], sizeof(uint32_t), 0);
        shift_global[block_idx] |= bit_stream[thread_idx] >> (bit_length - shift_count);
        __threadfence();
        shift_global[block_idx] = SET_HIGH_BIT(shift_global[block_idx], sizeof(uint32_t));
    }

    if(threadIdx.x == 0 && blockIdx.x != 0){
        //block inter
        WAIT(shift_global[blockIdx.x], sizeof(uint32_t), 1);
        shift_block[0] = shift_global[blockIdx.x];
        shift_global[blockIdx.x] = CLEAR_HIGH_BIT(shift_global[blockIdx.x], sizeof(uint32_t));
    }

    WAIT(shift_block[threadIdx.x], sizeof(uint32_t), 1);
    shift_block[threadIdx.x] = CLEAR_HIGH_BIT(shift_block[threadIdx.x], sizeof(uint32_t));
    bit_stream[thread_idx] = (bit_stream[thread_idx] << shift_count) | shift_block[threadIdx.x];

}

void get_bit_stream(uint32_t *bit_stream, int size){
    for(int i = 0; i < size; i++){
        bit_stream[i] = SET_HIGH_BIT(bit_stream[i], sizeof(uint32_t));
    }
}

int main()
{
    int iteration_time = GRID_NUM;
    int grid_dim = BLOCK_NUMBER * THREAD_NUM_PER_BLOCK;
    int bit_stream_size = GRID_NUM * BLOCK_NUMBER * THREAD_NUM_PER_BLOCK;
    int shift_count = 1;
    uint32_t bit_stream[bit_stream_size] = {0, 1, 2, 4, 8, 16, 32, 64};
    uint32_t bit_stream_shift[bit_stream_size] = {0, 2, 4, 8, 16, 32, 64, 128};
    uint32_t *bit_stream_gpu;
    uint32_t shift_global[BLOCK_NUMBER] = {0};
    uint32_t *shift_global_gpu;

    cudaMalloc((void **)&bit_stream_gpu, sizeof(uint32_t)*grid_dim);
    cudaMalloc((void **)&shift_global_gpu, sizeof(uint32_t)*BLOCK_NUMBER);

    shift_global[0] = SET_HIGH_BIT(shift_global[0], sizeof(uint32_t));
    cudaMemcpy(shift_global_gpu, shift_global, sizeof(uint32_t)*BLOCK_NUMBER, cudaMemcpyHostToDevice);
    get_bit_stream(bit_stream, bit_stream_size);

    for(int i = 0; i < iteration_time; i++){
        cudaMemcpy(bit_stream_gpu, bit_stream+i*grid_dim,
            grid_dim*sizeof(uint32_t), cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(THREAD_NUM_PER_BLOCK,1,1);
        shift_left<<<BLOCK_NUMBER, threadsPerBlock>>>(bit_stream_gpu, sizeof(uint32_t), shift_count, shift_global_gpu);

        cudaMemcpy(bit_stream+i*grid_dim, bit_stream_gpu,
            grid_dim*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    for (int j = 0; j < 8; j++){
        printf("host_bit_stream:%u\n",bit_stream[j]);
    }

    for(int i = 0; i < bit_stream_size; i++){
        if(CLEAR_LOW_BIT(bit_stream[i]) == bit_stream_shift[i]){
            printf("[%d]: {bit_stream:%u, shift_global_stream:%u}\n", i,
                bit_stream[i] >> shift_count, bit_stream_shift[i]);
        }
        else{
            printf("ERROR[%d]: {bit_stream:%u, shift_global_stream:%u}\n", i,
                bit_stream[i] >> shift_count, bit_stream_shift[i]);
        }
    }

    cudaFree(shift_global_gpu);
    cudaFree(bit_stream_gpu);
    return 0;
}
