#include <stdio.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

__device__ uint32_t mark_seq_bits_left(uint32_t stream){

}

__device__ uint32_t mark_seq_bits_right(uint32_t stream){

}

__device__ uint32_t matchstar(uint32_t carry, uint32_t propagate){

}

__device__ uint32_t get_carry_bit(uint32_t op1, uint32_t op2, uint32_t sum, int shift_count){

}

__device__ uint32_t get_propagate_bit(uint32_t propagate){

}

__device__ extract_lowest_bit(uint32_t num){

}

__device__ remove_lowest_bit(uint32_t num){
    
}

__device__ uint32_t flag_odd_global;

/** determine whether a sequential same character in a long string is even or odd. 
 ** inner-block version
*/
__global__ count_seq_bit(uint32_t * stream, uint32_t size) {
    __shared__ uint32_t bitmap_odd[WARP_SIZE]; // odd and propagate
    __shared__ uint32_t bitmap_odd_pre[WARP_SIZE];
    __shared__ uint32_t bitmap_propagate[WARP_SIZE];
    __shared__ uint32_t carry = 0, propagate = 0;
    uint32_t tid = threadIdx.x;
    //1. mark the most left position in sequential 1'b if its number is odd
    uint32_t left_seq_bit = mark_seq_bits_left(stream[tid]);
    
    //2. mark the most right position in sequential 1'b if its number is odd
    uint32_t right_seq_bit = mark_seq_bits_right(stream[tid]);
    
    //3. propagate the odd bit between different stream
    int bitmap_odd_index = ((tid+1) % size) / 32;
    int bitmap_odd_offset = ((tid+1) % size) % 32;
    int bitmap_propagate_index = tid / 32;
    int bitmap_propagate_offset = tid % 32;

    uint32_t flag_odd = RIGHT_SHIFT_BIT(left_seq_bit, BYTE_SIZE*sizeof(uint32_t)-1);
    uint32_t flag_propagate = stream[tid] == PROPAGATE_MASK(BYTE_SIZE*sizeof(uint32_t)) ? 1 : 0;
    atomicOr(bitmap_odd[bitmap_odd_index],  LEFT_SHIFT_BIT(flag_odd) << bitmap_odd_offset);
    atomicOr(bitmap_propagate[bitmap_propagate_index], LEFT_SHIFT_BIT(flag_propagate) << bitmap_propagate_offset);
    __syncthreads();

    if(tid == 0){
        uint32_t flag_odd_tmp = RIGHT_SHIFT_BIT(bitmap[tid], 0);
        bitmap_odd[tid] |= flag_odd_global;
        flag_odd_global = flag_odd_tmp;
    }

    // flag_odd + flag_propagate
    int base = (size + sizeof(int) * BYTE_SIZE - 1) / (sizeof(uint32_t) * BYTE_SIZE);
    assert(base <= 32);
    int shift_count;
    uint32_t carry_bit, propagate_bit;
    if(tid < WARP_SIZE) {
        bitmap_odd[tid] = bitmap_odd[tid] + bitmap_propagate[tid];
        shift_count = sizeof(uint32_t) * BYTE_SIZE - 1;
        carry_bit = get_carry_bit(bitmap_odd[tid], bitmap_propagate[tid], sum, shift_count);
        propagate_bit = get_propagate_bit(bitmap_propagate[tid]);
        atomicOr(carry, LEFT_SHIFT_BIT(carry_bit, (tid + 1) % WARP_SIZE));
        atomicOr(propagate, LEFT_SHIFT_BIT(propagate_bit, tid));
    }
    __syncthreads();
    int index, offset;
    if(tid == 0){
        index = size / 32;
        offset = size % 32;
        flag_odd_global |= RIGHT_SHIFT_BIT(carry, (index+1)%WARP_SIZE);
        carry &= ~(1U << ((index+1)%WARP_SIZE));
        flag_odd_global |= offset != 31 ? RIGHT_SHIFT_BIT(bitmap_odd[index], offset+1) : 0;
        bitmap_odd[index] &= offset != 31 ? ~(1U << offset+1) : 0xFFFFFFFFU;

        carry_bit = matchstar(carry, propagate);
        flag_odd_global |= carry_bit;
    }
    uint32_t bitmap_odd_tmp;
    if(tid < WARP_SIZE) {
        carry_bit = RIGHT_SHIFT_BIT(carry, tid);
        bitmap_odd[tid] += carry_bit;
        bitmap_odd_tmp &= bitmap_odd[tid] & bitmap_propagate[tid];
        bitmap_odd[tid] ^= bitmap_odd_tmp;
    }

    //4. correct the marker in the lowest bit in left_seq_bit.
    uint32_t lowest_right_seq_bit = RIGHT_SHIFT_BIT(right_seq_bit, 0);
    uint32_t lowest_stream_bit = RIGHT_SHIFT_BIT(stream[tid], 0);
    index = tid / 32;
    offset = tid % 32;
    flag_odd = RIGHT_SHIFT_BIT(bitmap_odd[index], offset);
    if(lowest_right_seq_bit > lowest_right_seq_bit ^ flag_odd){
        remove_lowest_bit(left_seq_bit);
    }else if(lowest_right_seq_bit < lowest_right_seq_bit_ ^ flag_odd){
        if(lowest_stream_bit == 1U){
            left_seq_bit ^= extract_lowest_bit(stream + 1) >> 1; 
        }else{
            index = (tid-1) / 32;
            offset = (tid-1) % 32;
            atomicOr(bitmap_odd_pre[index], LEFT_SHIFT_BIT(1, offset)); 
        }
    }
    __syncthreads();

    index = tid / 32;
    offset = tid % 32;
    flag_odd = RIGHT_SHIFT_BIT(bitmap_odd_pre[index], offset);
    left_seq_bit |= LEFT_SHIFT_BIT(flag_odd, 31);
    
    stream[tid] = left_seq_bit;
}