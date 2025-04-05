#include <stdio.h>
#include <cuda_runtime.h>

void bit_stream_add(uint32_t *stream1, uint32_t *stream2, uint32_t *result, size_t size){
    uint32_t carry = 0;
    for (size_t i = 0; i < n; i++) {
        __uint64_t sum = (__uint64_t)a[i] + b[i] + carry;
        res[i] = (uint32_t)sum;
        carry = (uint32_t)(sum >> 32);
    }
    if (carry) {
        res[n] = carry;
    }
}

/* Kogge-Stone:
 * 1.Generate carry (G) and propagate carry (P)
 * G(i) = A(i)B(i)
 * P(i) = A(i)^B(i)
 * 2. carry bit
 * C(i) = C(i-1) + P(i-1)C(i-1)
 * S(i) = P(i)^C(i)
*/ 
__global__ void bit_stream_add1_parallel(uint32_t *stream1, uint32_t *stream2,
    uint32_t *result, size_t size){

}

#define GRID_NUM 2
#define BLOCK_NUM 2
#define THREAD_NUM_PER_BLOCK 2
#define LEFT_SHIFT_BIT(num, count) (num << count)
#define RIGHT_SHIFT_BIT(num, count) ((num >> count) & 0x00000001)
#define PROPAGATE_MASK(count) (1 << count - 1) 

/*MatchStar: 
 *
 */
 __global__ void bit_stream_add2_parallel(uint32_t *stream1, uint32_t *stream2,
    uint32_t carry_global, uint32_t propagate_global, uint32_t * result, size_t size){
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bid = blockIdx.x;
    __shared__ uint32_t carry_block = 0;
    __shared__ uint32_t propagate_block = 0;
    uint32_t sum = stream1[tid] + stream2[tid];

    //generate the G_bit and P_bit
    propagate_block |= sum == (uint32_t)PROPAGATE_MASK(sizeof(uint32_t)) ? LEFT_SHIFT_BIT(1, threadIdx.x) : 0;
    carry_tmp = (RIGHT_SHIFT_BIT(stream1[tid], sizeof(uint32_t)) &
                RIGHT_SHIFT_BIT(stream2[tid], sizeof(uint32_t)-1)) | (
                RIGHT_SHIFT_BIT(stream1[tid], sizeof(uint32_t)-1) ^
                RIGHT_SHIFT_BIT(stream2[tid], sizeof(uint32_t)-1) &
                (!RIGHT_SHIFT_BIT(sum, sizeof(uint32_t)-1)));    
    // global
    if(bid == 0 & tid == 0){
        carry_block |= carry_global;
        carry_global &= 0x7FFFFFFF;
        // free global;
    }
    // block
    if(threadIdx.x < blockDim.x-1){
        carry_block |= LEFT_SHIFT_BIT(carry_tmp, threadIdx.x + 1);
    }
    // global
    if(threadIdx.x == blockDim.x-1){
        // stop if carry bit not free
        carry_global | = LEFT_SHIFT_BIT(carry_tmp, (bid+1)%gridDim.x);
    }

    //move carry to block
    if(bid > 0){
        // stop if carry not to be set
        carry_block |= RIGHT_SHIFT_BIT(carry_global, bid%gridDim.x);
        carry_global &= ~((uint32_t)LEFT_SHIFT_BIT(1, bid%gridDim.x));
    }
    __syncthreads();

    if(threadIdx.x == 0){
        //1. update carry_block
        //pre-matchstar
        //G_bit + P_bit
        uint32_t carry_GP_tmp;
        uint32_t sum_GP = carry_block + propagate_block;
        propagate_global = sum_GP == (uint32_t)PROPAGATE_MASK(gridDim.x) | LEFT_SHIFT_BIT(1, bid) : 0;

        carry_GP_tmp = (RIGHT_SHIFT_BIT(carry_block, blockDim.x-1) &
                        RIGHT_SHIFT_BIT(propagate_block, blockDim.x-1)) | (
                        RIGHT_SHIFT_BIT(carry_block, blockDim.x-1) ^
                        RIGHT_SHIFT_BIT(propagate_block, blockDim.x-1) &
                        (!RIGHT_SHIFT_BIT(sum_GP, blockDim.x-1)));
        carry_global |= LEFT_SHIFT_BIT(carry_GP_tmp, bid+1%gridDim.x);
        // update carry_global.matchstar for (carry_global & propagate_blobal), the 0 position don't participate in this process.
        // correct the sum_GP except block 0 
        // update the carry_block: post-matchstar
    }
    // correct the sum
    

 }
void fill_array(uint32_t* stream, int size);

__device__ uint32_t carry_global, propagate_global;

int main(){
    uint32_t *stream1, *stream2, *result;
    uint32_t *d_stream1, * d_stream2, *d_result;

    // alloc space and initialize array
    int size = GRID_NUM * BLOCK_NUM * THREAD_NUM_PER_BLOCK;
    stream1 = (uint32_t *)malloc(size * sizeof(uint32_t));
    fill_array(stream1, size);
    stream2 = (uint32_t *)malloc(size * sizeof(uint32_t));
    fill_array(stream2, size);
    result = (uint32_t *)malloc(size * sizeof(uint32_t));

    // copy data from host to device
    int d_size = BLOCK_NUM * THREAD_NUM_PER_BLOCK;
    cudaMalloc((void **)&d_stream1, d_size*sizeof(uint32_t));
    cudaMalloc((void **)&d_stream2, d_size*sizeof(uint32_t));
    cudaMalloc((void **)&d_result, d_size*sizeof(uint32_t));

    uint32_t initial_value = 0;
    cudaMemcpyToSymbol(carry_global, &initial_value, sizeof(uint32_t));
    cudaMemcpyToSymbol(propagate_global, &initial_value, sizeof(uint32_t));

    for(int i = 0; i < GRID_NUM; i++){
        cudaMemcpy(d_stream1, stream1, d_size*sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_stream2, stream2, d_size*sizeof(uint32_t), cudaMemcpyHostToDevice);

        bit_stream_add2_parallel(d_stream1, d_stream2, carry_global, propagate_global, d_result, d_size);

        cudaMemcpy(result+i*d_size, d_result, d_size*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }

    for(int i = 0; i < size; i++){
        printf("[%d]: %u + %u = %u", i, stream1[i], stream2[i], result);
    }
    return 0;
}