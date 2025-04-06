#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <cuda_runtime.h>

void bit_stream_add(uint32_t *stream1, uint32_t *stream2, uint32_t *result, size_t size){
    uint32_t carry = 0;
    for (size_t i = 0; i < size; i++) {
        __uint64_t sum = (__uint64_t)stream1[i] + stream2[i] + carry;
        result[i] = (uint32_t)sum;
        carry = (uint32_t)(sum >> 32);
    }
    if (carry) {
        result[size] = carry;
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

#define GRID_NUM 1
#define BLOCK_NUM 1
#define THREAD_NUM_PER_BLOCK 4
#define LEFT_SHIFT_BIT(num, count) (num << count)
#define RIGHT_SHIFT_BIT(num, count) ((num >> count) & 0x00000001)
#define PROPAGATE_MASK(count) ((1 << count) - 1)
#define SET(num, index) num = (num | 1 << index)
#define UNSET(num, index) num = (num & (~(1 << index)))
#define CHECK_LOCK(num, index) (num & (1 << index))
#define WAIT(num, index, target) while((CHECK_LOCK(__ldg(&num), index) >> index) != target)
#define WAIT_GLOBAL(num, count, target) while((__ldg(&num) & ((1 << count) - 1)) != target) \
{ printf("wait:"); printf("(num:%u",num); printf("count:%u", count); printf("%u\n)", target);}

__device__ uint32_t carry_global, propagate_global, flag_global;
__device__ int carry_global_lock, propagate_global_lock, flag_global_lock;
__device__ void lock(int *mutex) {
    printf("muxte:%d\n",*mutex);
    while (atomicCAS(mutex, 0, 1) != 0){ printf("muxte_while:%d\n",*mutex);};
    printf("muxte2:%d\n",*mutex);

}
__device__ void unlock(int *mutex) {
    atomicExch(mutex, 0);
}
/*MatchStar: 
 *
 */
__global__ void bit_stream_add2_parallel(uint32_t *stream1, uint32_t *stream2,
    uint32_t * result, size_t size){
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t bid = blockIdx.x;
    __shared__ uint32_t carry_block;
    __shared__ uint32_t propagate_block;
    __shared__ int carry_lock, propagate_lock;
    result[tid] = stream1[tid] + stream2[tid];

    if (threadIdx.x == 0) {
        carry_block = 0;
        propagate_block = 0;
        carry_lock = 0;
        propagate_lock = 0;
    }
    __syncthreads();

    //generate the G_bit and P_bit
    printf("propa_lock:%d\n", propagate_lock);
    lock(&propagate_lock);
    propagate_block |= result[tid] == (uint32_t)0xFFFFFFFF ? LEFT_SHIFT_BIT(1, threadIdx.x) : 0;
    unlock(&propagate_lock);
    printf("propa_lock2:%d\n", propagate_lock);

    uint32_t carry_block_tmp = 0;
    carry_block_tmp = (RIGHT_SHIFT_BIT(stream1[tid], 8*sizeof(uint32_t)-1) &
                RIGHT_SHIFT_BIT(stream2[tid], 8*sizeof(uint32_t)-1)) | (
                RIGHT_SHIFT_BIT(stream1[tid], 8*sizeof(uint32_t)-1) ^
                RIGHT_SHIFT_BIT(stream2[tid], 8*sizeof(uint32_t)-1) &
                (!RIGHT_SHIFT_BIT(result[tid], 8*sizeof(uint32_t)-1)));  
    printf("carry_block_tmp:%u\n", carry_block_tmp); 
    // global
    if(bid == 0 & tid == 0){
        assert(CHECK_LOCK(flag_global, 0) != 0);
        lock(&carry_lock);
        carry_block |= carry_global;
        unlock(&carry_lock);

        lock(&carry_global_lock);
        carry_global = (carry_global >> 1) << 1;
        unlock(&carry_global_lock);

        __threadfence();

        lock(&flag_global_lock);
        UNSET(flag_global, 0);
        unlock(&flag_global_lock);

        printf("carry_block_0_0: %u\n", carry_block);
    }
    // block
    if(threadIdx.x < blockDim.x-1){
        lock(&carry_lock);
        carry_block |= LEFT_SHIFT_BIT(carry_block_tmp, threadIdx.x + 1);
        unlock(&carry_lock);
        printf("carry_block_x_1-: %u\n", carry_block);
    }
    // global
    if(threadIdx.x == (uint32_t)(blockDim.x-1)){
        WAIT(flag_global, (bid+1)%gridDim.x, 0);
        lock(&carry_global_lock);
        carry_global |= LEFT_SHIFT_BIT(carry_block_tmp, (bid+1)%gridDim.x);
        unlock(&carry_global_lock);
        __threadfence();
        lock(&flag_global_lock);
        SET(flag_global, (bid+1)%gridDim.x);
        unlock(&flag_global_lock);
    }

    //move carry global to block
    if(bid > 0 && threadIdx.x == 0){
        WAIT(flag_global, bid, (uint32_t)1);
        lock(&carry_lock);
        carry_block |= RIGHT_SHIFT_BIT(carry_global, bid);
        unlock(&carry_lock);
        lock(&carry_global_lock);
        carry_global &= ~((uint32_t)LEFT_SHIFT_BIT(1, bid));
        unlock(&carry_global_lock);
        lock(&flag_global_lock);
        UNSET(flag_global, bid);
        unlock(&flag_global_lock);
        printf("carry_block_1-_0: %u\n", carry_block);
    } 
    __syncthreads();
    
    if((bid == 0) && (threadIdx.x == 0)){
        WAIT(flag_global, bid, (uint32_t)1);
        lock(&flag_global_lock);
        UNSET(flag_global, bid);
        unlock(&flag_global_lock);
    }

    //1. update carry_block
    if(threadIdx.x == 0){
        //pre-matchstar
        printf("carry_block_pre:%u\n", carry_block);
        uint32_t tmp1 = carry_block & propagate_block;

        //G_bit + P_bit
        uint32_t carry_global_tmp;
        tmp1 = tmp1 + propagate_block;
        lock(&propagate_global_lock);
        propagate_global = tmp1 == (uint32_t)PROPAGATE_MASK(gridDim.x) ? LEFT_SHIFT_BIT(1, bid) : 0;
        unlock(&propagate_global_lock);
        // set propagate_global
        carry_global_tmp = (RIGHT_SHIFT_BIT(carry_block, blockDim.x-1) &
                        RIGHT_SHIFT_BIT(propagate_block, blockDim.x-1)) | (
                        RIGHT_SHIFT_BIT(carry_block, blockDim.x-1) ^
                        RIGHT_SHIFT_BIT(propagate_block, blockDim.x-1) &
                        (!RIGHT_SHIFT_BIT(tmp1, blockDim.x-1)));
        WAIT(flag_global, (bid+1)%gridDim.x, 0);
        lock(&carry_global_lock);
        carry_global |= LEFT_SHIFT_BIT(carry_global_tmp, (bid+1)%gridDim.x);
        unlock(&carry_global_lock);
        lock(&flag_global_lock);
        SET(flag_global, (bid+1)%gridDim.x);
        unlock(&flag_global_lock);

        // matchstar:update carry_global
        printf("flag_global:%u\n", flag_global);
        WAIT_GLOBAL(flag_global, gridDim.x, (1<<gridDim.x)-1);
        uint32_t tmp2 = ((carry_global >> 1) << 1) & propagate_global;
        tmp2 += propagate_global;
        tmp2 ^= propagate_global;
        tmp2 &= (carry_global >> 1) << 1;

        carry_global_tmp = (RIGHT_SHIFT_BIT(carry_global, blockDim.x-1) &
                        RIGHT_SHIFT_BIT(propagate_global, blockDim.x-1)) | (
                        RIGHT_SHIFT_BIT(carry_global, blockDim.x-1) ^
                        RIGHT_SHIFT_BIT(propagate_global, blockDim.x-1) &
                        (!RIGHT_SHIFT_BIT(tmp2, blockDim.x-1)));

        atomicOr(&carry_global, LEFT_SHIFT_BIT(carry_global_tmp, 0));
        // correct the tmp1 except block 0 
        if(bid > 0){
            tmp1 += RIGHT_SHIFT_BIT(tmp2, bid); 
        }
        // update the carry_block: post-matchstar
        tmp1 ^= propagate_block;
        carry_block &= tmp1;
        printf("carry_block:%u\n", carry_block);
    }
    __syncthreads();
    //2.correct the result[tid]
    result[tid] += RIGHT_SHIFT_BIT(carry_block, threadIdx.x);
}
void fill_array(uint32_t* stream, int size){
    for(int i = 0; i < size; i++){
        stream[i] = i;
        stream[i] |= (uint32_t)(1 << 31);
    }
}

int main(){
    uint32_t *stream1, *stream2, *result;
    uint32_t *d_stream1, * d_stream2, *d_result;

    // alloc space and initialize array
    int size = GRID_NUM * BLOCK_NUM * THREAD_NUM_PER_BLOCK;
    stream1 = (uint32_t *)malloc(size * sizeof(uint32_t));
    fill_array(stream1, size);
    stream2 = (uint32_t *)malloc(size * sizeof(uint32_t));
    fill_array(stream2, size);
    result = (uint32_t *)malloc((size+1) * sizeof(uint32_t));

    // copy data from host to device
    int d_size = BLOCK_NUM * THREAD_NUM_PER_BLOCK;
    cudaMalloc((void **)&d_stream1, d_size*sizeof(uint32_t));
    cudaMalloc((void **)&d_stream2, d_size*sizeof(uint32_t));
    cudaMalloc((void **)&d_result, d_size*sizeof(uint32_t));

    uint32_t initial_value = 0;
    cudaMemcpyToSymbol(carry_global, &initial_value, sizeof(uint32_t));
    cudaMemcpyToSymbol(propagate_global, &initial_value, sizeof(uint32_t));
    cudaMemcpyToSymbol(carry_global_lock, &initial_value, sizeof(int));
    cudaMemcpyToSymbol(propagate_global_lock, &initial_value, sizeof(int));
    cudaMemcpyToSymbol(flag_global_lock, &initial_value, sizeof(int));

    for(int i = 0; i < GRID_NUM; i++){
        SET(initial_value, 0);
        cudaMemcpyToSymbol(flag_global, &initial_value, sizeof(uint32_t));
        cudaMemcpy(d_stream1, stream1, d_size*sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_stream2, stream2, d_size*sizeof(uint32_t), cudaMemcpyHostToDevice);

        bit_stream_add2_parallel<<<BLOCK_NUM, THREAD_NUM_PER_BLOCK>>>(d_stream1, d_stream2,
            d_result, d_size);

        cudaMemcpy(result+i*d_size, d_result, d_size*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&initial_value, flag_global, sizeof(uint32_t));
        assert((initial_value & 0x01) != 0);
        initial_value = 0;
    }
    cudaMemcpyFromSymbol(&result[size], carry_global, sizeof(uint32_t));
    result[size] = RIGHT_SHIFT_BIT(result[size], 0);

    for(int i = 0; i < size; i++){
        printf("[%d]: %u + %u = %u\n", i, stream1[i], stream2[i], result[i]);
    }
    printf("carry:%u\n", result[size]);
    return 0;
}