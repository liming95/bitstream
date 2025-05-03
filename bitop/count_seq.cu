#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <cuda_runtime.h>

#define GRID_NUM 1
#define BLOCK_NUM 1
#define THREAD_NUM_PER_BLOCK 4
#define BYTE_SIZE 8
#define WARP_SIZE 32
#define CLEAR_LOWEST_BIT_MASK 0xFFFFFFFE
#define LEFT_SHIFT_BIT(num, count) (num << count)
#define RIGHT_SHIFT_BIT(num, count) ((num >> count) & 0x00000001)
#define PROPAGATE_MASK(N) ((N) >= 32 ? 0xFFFFFFFFU : ((1U << (N)) - 1U))

__device__ uint32_t mark_seq_bits_left(uint32_t stream){
    uint32_t stream_tmp, result = 0;
    while(stream != 0){
        stream_tmp = stream >> 1;
        stream_tmp ^= stream;
        stream_tmp &= stream;
        result ^= stream_tmp;
        stream ^= stream_tmp;
        stream = stream << 1;
    }
    return result;
}

__device__ uint32_t mark_seq_bits_right(uint32_t stream){
    uint32_t stream_tmp, result = 0;
    while(stream != 0){
        stream_tmp = stream << 1;
        stream_tmp ^= stream;
        stream_tmp &= stream;
        result ^= stream_tmp;
        stream ^= stream_tmp;
        stream = stream >> 1;
    }
    return result;
}

__device__ uint32_t get_carry_bit(uint32_t op1, uint32_t op2, uint32_t sum, int shift_count){
    uint32_t cond1, cond2;
    uint32_t op1_highest_bit = RIGHT_SHIFT_BIT(op1, shift_count);
    uint32_t op2_highest_bit = RIGHT_SHIFT_BIT(op2, shift_count);
    uint32_t sum_highest_bit = RIGHT_SHIFT_BIT(sum, shift_count);

    cond1 = op1_highest_bit & op2_highest_bit;
    cond2 = (op1_highest_bit ^ op2_highest_bit) & (!sum_highest_bit);
    return cond1 | cond2;
}

__device__ uint32_t matchstar(uint32_t *carry_p, uint32_t propagate, int size){
    uint32_t carry = *carry_p;
    uint32_t carry_bit = 0;
    carry &= propagate;
    carry += propagate;
    //carry_bit = get_carry_bit(*carry_p, propagate, carry, size-1);
    carry ^= propagate;
    *carry_p |= carry;
    carry_bit = (*carry_p) >> size;
    *carry_p &= ~(1U << size);
    return carry_bit;
}

__device__ uint32_t get_propagate_bit(uint32_t propagate){
    return propagate == 0xFFFFFFFFU ? 1U : 0U;
}

__device__ uint32_t extract_lowest_bit(uint32_t num){
    uint32_t num_tmp = num;
    num = num - 1;
    num ^= num_tmp;
    num += 1;
    num = num >> 1;
    return num;
}

__device__ uint32_t remove_lowest_bit(uint32_t num){
    num = ((num - 1) & num);
    return num;
}

__device__ uint32_t flag_odd_global, flag_odd_global_pre;

/** determine whether a sequential same character in a long string is even or odd.
 ** inner-block version
*/
__global__ void count_seq_bit(uint32_t * stream, uint32_t size) {
    __shared__ uint32_t bitmap_odd[WARP_SIZE]; // odd and propagate
    __shared__ uint32_t bitmap_propagate[WARP_SIZE];
    __shared__ uint32_t carry, propagate;
    uint32_t tid = threadIdx.x;

    if (threadIdx.x == 0) {
        carry = 0;
        propagate = 0;
    }
    if (tid < WARP_SIZE){
        bitmap_odd[tid] = 0;
        bitmap_propagate[tid] = 0;
    }
    __syncthreads();

    //1. mark the most left position in sequential 1'b if its number is odd
    uint32_t left_seq_bit = mark_seq_bits_left(stream[tid]);

    //2. mark the most right position in sequential 1'b if its number is odd
    uint32_t right_seq_bit = mark_seq_bits_right(stream[tid]);
    printf("tid[%u]: left: 0x%08x; right: 0x%08x\n", tid, left_seq_bit, right_seq_bit);

    //3. propagate the odd bit between different stream
    int bitmap_odd_index = ((tid+1) % size) / 32;
    int bitmap_odd_offset = ((tid+1) % size) % 32;
    int bitmap_propagate_index = tid / 32;
    int bitmap_propagate_offset = tid % 32;

    uint32_t flag_odd = RIGHT_SHIFT_BIT(left_seq_bit, BYTE_SIZE*sizeof(uint32_t)-1);
    uint32_t flag_propagate = stream[tid] == PROPAGATE_MASK(BYTE_SIZE*sizeof(uint32_t)) ? 1 : 0;
    atomicOr(&bitmap_odd[bitmap_odd_index],  LEFT_SHIFT_BIT(flag_odd, bitmap_odd_offset));
    atomicOr(&bitmap_propagate[bitmap_propagate_index], LEFT_SHIFT_BIT(flag_propagate, bitmap_propagate_offset));
    __syncthreads();

    if(tid == 0){
        uint32_t flag_odd_tmp = RIGHT_SHIFT_BIT(bitmap_odd[tid], 0);
        bitmap_odd[tid] &= CLEAR_LOWEST_BIT_MASK;
        bitmap_odd[tid] |= flag_odd_global;
        flag_odd_global = flag_odd_tmp;
    }
    // Todo: why the bitmap_odd value are different in differnt thread.
    printf("tid[%u(3)]: bitmap{odd: 0x%08x, propagate: 0x%08x}\n", tid, bitmap_odd[0], bitmap_propagate[0]);

    // flag_odd + flag_propagate
    int base = (size + sizeof(int) * BYTE_SIZE - 1) / (sizeof(uint32_t) * BYTE_SIZE);
    assert(base <= 32);
    int shift_count;
    uint32_t carry_bit, propagate_bit;
    uint32_t tmp;
    if(tid < WARP_SIZE) {
        tmp = bitmap_odd[tid];
        bitmap_odd[tid] = bitmap_odd[tid] + bitmap_propagate[tid];
        shift_count = sizeof(uint32_t) * BYTE_SIZE - 1;
        carry_bit = get_carry_bit(tmp, bitmap_propagate[tid], bitmap_odd[tid], shift_count);
        propagate_bit = get_propagate_bit(bitmap_propagate[tid]);
        atomicOr(&carry, LEFT_SHIFT_BIT(carry_bit, (tid + 1) % WARP_SIZE));
        atomicOr(&propagate, LEFT_SHIFT_BIT(propagate_bit, tid));
    }
    __syncthreads();
    int index, offset;
    if(tid == 0){
        index = size / 32;
        offset = size % 32;
        flag_odd_global |= RIGHT_SHIFT_BIT(carry, (index+1)%WARP_SIZE);
        carry &= ~(1U << ((index+1)%WARP_SIZE));
        flag_odd_global |= offset != 31 ? RIGHT_SHIFT_BIT(bitmap_odd[index], offset+1) : 0;
        bitmap_odd[index] &= offset != 31 ? ~(1U << (offset+1)) : 0xFFFFFFFFU;

        carry_bit = matchstar(&carry, propagate, index+1);
        flag_odd_global |= carry_bit;
    }
    // uint32_t bitmap_odd_tmp;
    if(tid < WARP_SIZE) {
        carry_bit = RIGHT_SHIFT_BIT(carry, tid);
        bitmap_odd[tid] += carry_bit;
        // bitmap_odd_tmp &= bitmap_odd[tid] & bitmap_propagate[tid];
        // bitmap_odd[tid] ^= bitmap_odd_tmp;
    }
    printf("tid[%u]: bitmap_odd + bitmap_propagate: 0x%08x. carry: 0x%08x, propagate:0x%08x\n", tid,
        bitmap_odd[tid], carry, propagate);

    //4. correct the marker in the lowest bit in left_seq_bit.
    uint32_t lowest_right_seq_bit = RIGHT_SHIFT_BIT(right_seq_bit, 0);
    uint32_t lowest_stream_bit = RIGHT_SHIFT_BIT(stream[tid], 0);
    index = tid / 32;
    offset = tid % 32;
    // Todo: flag_odd << ? ^ left_seq_bit
    flag_odd = RIGHT_SHIFT_BIT(bitmap_odd[index], offset);
    if(lowest_right_seq_bit == 1U && flag_odd == 1U){
        left_seq_bit = remove_lowest_bit(left_seq_bit);
        atomicAnd(&bitmap_odd[index], ~(LEFT_SHIFT_BIT(1, offset)));
    }else if(lowest_right_seq_bit == 0U && flag_odd == 1U){
        if(lowest_stream_bit == 1U){
            left_seq_bit ^= extract_lowest_bit(stream[tid] + 1) >> 1;
            atomicAnd(&bitmap_odd[index], ~(LEFT_SHIFT_BIT(1, offset)));
        }
    }
    __syncthreads();
    printf("tid[%d(4)]: bitmap_odd:0x%08x, left_seq_bit:0x%08x\n", tid, bitmap_odd[tid], left_seq_bit);

    // Todo: clean this odd information when the odd information is propagated
    if(tid < size-1){
        index = (tid+1) / 32;
        offset = (tid+1) % 32;
        flag_odd = RIGHT_SHIFT_BIT(bitmap_odd[index], offset);
        left_seq_bit &= 0x7FFFFFFFU;
        left_seq_bit |= LEFT_SHIFT_BIT(flag_odd, 31);

        stream[tid] = left_seq_bit;
    } else{
        // If the flag_odd_global from pre block doesn't be consumed, this value need to be returned to the pre block.
        left_seq_bit &= 0x7FFFFFFFU;
        stream[tid] = left_seq_bit;
        flag_odd_global_pre = RIGHT_SHIFT_BIT(bitmap_odd[0], 0);
    }
}
void fill_array(uint32_t *stream, int size){
    stream[0] = 0x80FF0001U;    //0x01
    stream[1] = 0xFFFFFFFFU;    //0x00
    stream[2] = 0x8000001FU;    //0x80000000U
    stream[3] = 0x8000000FU;    //0x00
    for(int i = 4; i < size; i++){
        stream[i] = 0U;
    }
}
int main(){
    uint32_t *stream;
    uint32_t *d_stream;

    // alloc space and initialize array
    int size = GRID_NUM * BLOCK_NUM * THREAD_NUM_PER_BLOCK;
    stream = (uint32_t *)malloc(size * sizeof(uint32_t));
    fill_array(stream, size);
    for(int i = 0; i < size; i++){
        printf("[%d]: 0x%08x\n", i, stream[i]);
    }
    // copy data from host to device
    int d_size = BLOCK_NUM * THREAD_NUM_PER_BLOCK;
    cudaMalloc((void **)&d_stream, d_size*sizeof(uint32_t));

    uint32_t flag_odd = 1, flag_odd_pre;
    cudaMemcpyToSymbol(flag_odd_global, &flag_odd, sizeof(uint32_t));
    cudaMemcpyToSymbol(flag_odd_global_pre, &flag_odd_pre, sizeof(uint32_t));

    for(int i = 0; i < GRID_NUM; i++){
        cudaMemcpy(d_stream, stream+i*d_size, d_size*sizeof(uint32_t), cudaMemcpyHostToDevice);

        count_seq_bit<<<BLOCK_NUM, THREAD_NUM_PER_BLOCK>>>(d_stream, d_size);

        cudaMemcpy(stream+i*d_size, d_stream, d_size*sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&flag_odd, flag_odd_global, sizeof(uint32_t));
        cudaMemcpyFromSymbol(&flag_odd_pre, flag_odd_global_pre, sizeof(uint32_t));

        printf("flag_odd: %u\n", flag_odd);
        printf("flag_odd_pre: %u\n", flag_odd_pre);
    }

    for(int i = 0; i < size; i++){
        printf("[%d]: 0x%08x\n", i, stream[i]);
    }
    return 0;
}
