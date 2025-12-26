## parallel exectuion for carry operation
import numpy as np
# the operation in the block
def block_parallel_bitwise_and(marker1:np.array, marker2:np.array)->np.array:
    marker = marker1 & marker2
    return marker

def block_parallel_bitwise_or(marker1, marker2):
    marker = marker1 | marker2
    return marker

def block_parallel_bitwise_not(marker1):
    marker = ~marker1
    return marker

def block_parallel_bitwise_left_shift(marker1):
    carry_bit = (marker1 & 0x80) >> 7
    tmp_value = marker1 << 1
    # print(carry_bit, tmp_value)
    shifted_carry_bit = np.zeros_like(carry_bit)
    shifted_carry_bit[1:] = carry_bit[:-1]
    marker = tmp_value | shifted_carry_bit
    return marker

# the bitwise operation in different blocks
def multi_block_parallel_bitwise_and(marker1, marker2, block_size):
    data_size = len(marker1)
    iter_time = (data_size + block_size - 1) // block_size
    marker = np.zeros_like(marker1)
    for i in range(iter_time):
        start = i * block_size
        end = min((i+1)*block_size, data_size)
        marker[start:end] = block_parallel_bitwise_and(marker1[start:end], marker2[start:end])
    return marker

def multi_block_parallel_bitwise_or(marker1, marker2, block_size):
    data_size = len(marker1)
    iter_time = (data_size + block_size - 1) // block_size
    marker = np.zeros_like(marker1)
    for i in range(iter_time):
        start = i * block_size
        end = min((i+1)*block_size, data_size)
        marker[start:end] = block_parallel_bitwise_or(marker1[start:end], marker2[start:end])
    return marker

def multi_block_parallel_bitwise_not(marker1, block_size):
    data_size = len(marker1)
    iter_time = (data_size + block_size - 1) // block_size
    marker = np.zeros_like(marker1)
    for i in range(iter_time):
        start = i * block_size
        end = min((i+1)*block_size, data_size)
        marker[start:end] = block_parallel_bitwise_not(marker1[start:end])
    return marker

def multi_block_parallel_bitwise_left_shift(marker1, block_size):
    data_size = len(marker1)
    iter_time = (data_size + block_size - 1) // block_size
    marker = np.zeros_like(marker1)
    for i in range(iter_time):
        start = i * block_size
        end = min((i+1)*block_size, data_size)
        marker[start:end] = block_parallel_bitwise_left_shift(marker1[start:end])
    return marker

def extract_recalculation_data(marker_list, block_size, bit_len):
    # ele_bit_len = marker_list.dtype.itemsize * 8
    # data_len = (bit_len + ele_bit_len - 1) // ele_bit_len
    # data_size = marker_list.shape[1]
    # iter_time = (data_size + block_size - 1) // block_size

    # extract_marker_list = np.zeros((marker_list.shape[0], 2*data_len*(iter_time-1)), dtype=marker_list.dtype)

    # for marker, marker_tmp in zip(extract_marker_list,marker_list):
    #     marker[:data_len] = marker_tmp[block_size-data_len: block_size]
    #     for i in range(1,iter_time-1):
    #         start = i * block_size
    #         end = (i+1) * block_size
    #         start_m = i * data_len * 2 - data_len
    #         end_m = (i+1) * data_len * 2 - data_len
    #         marker[start_m: start_m+data_len] = marker_tmp[start:start+data_len]
    #         marker[end_m-data_len: end_m] = marker_tmp[end-data_len: end]

    #     start_m = (iter_time-1)*2*data_len-data_len
    #     start = (iter_time-1) * block_size
    #     final_data_len = min(data_len, data_size-start)
    #     marker[start_m:start_m+final_data_len] = marker_tmp[start:start+final_data_len]
    # print(extract_marker_list)
    # return extract_marker_list
    N, data_size = marker_list.shape
    ele_bit_len = marker_list.dtype.itemsize * 8
    data_len = (bit_len + ele_bit_len - 1) // ele_bit_len
    iter_time = (data_size + block_size - 1) // block_size

    out_len = 2 * data_len * (iter_time-1) #+ min(data_len, data_size - (iter_time-1)*block_size)
    extract_marker_list = np.zeros((N, out_len), dtype=marker_list.dtype)

    for idx in range(N):
        marker_tmp = marker_list[idx]
        marker_out = extract_marker_list[idx]

        first_len = min(data_len, block_size)
        marker_out[:first_len] = marker_tmp[block_size-first_len:block_size]

        for i in range(1, iter_time-1):
            start = i * block_size
            end = start + block_size
            start_m = (i-1)*2*data_len + data_len
            end_m = start_m + 2*data_len

            marker_out[start_m:start_m+data_len] = marker_tmp[start:start+data_len]
            marker_out[start_m+data_len:end_m] = marker_tmp[end-data_len:end]

        if iter_time > 1:
            start = (iter_time-1)*block_size
            final_len = min(data_len, data_size - start)
            start_m = (iter_time-2)*2*data_len + data_len
            marker_out[start_m:start_m+final_len] = marker_tmp[start:start+final_len]
    return extract_marker_list, 2*data_len

def correct_result(marker_res, marker_correct, block_size_res, block_size_correct):
    block_num_res = (len(marker_res) + block_size_res - 1) // block_size_res
    block_num_correct = (len(marker_correct) + block_size_correct - 1) // block_size_correct
    assert block_num_res == block_num_correct + 1
    data_len = block_size_correct // 2

    for i in range(block_num_correct):
        start = (i+1)*block_size_res
        start_m = i*block_size_correct+data_len
        marker_res[start:start+data_len] = marker_correct[start_m:start_m+data_len]

    return marker_res

def test_block_bitwise_op():
    org_marker1 = bytearray([0b11001100, 0b10101010])
    org_marker2 = bytearray([0b11110000, 0b00001111])
    marker1 = np.array(org_marker1)
    marker2 = np.array(org_marker2)
    expected_value_and = np.array([0b11000000, 0b00001010])
    print(f"marker dtype: {marker1.dtype}")
    marker = block_parallel_bitwise_and(marker1, marker2)
    assert (marker == expected_value_and).all(), f"the result is wrong with parallel and operation"

    expected_value_or = np.array([0b11111100, 0b10101111])
    marker = block_parallel_bitwise_or(marker1, marker2)
    assert (marker == expected_value_or).all(), f"the result is wrong with parallel or operation"

    expected_value_left_shift = np.array([0b10011000, 0b01010101])
    marker = block_parallel_bitwise_left_shift(marker1)
    # print(marker)
    assert (marker == expected_value_left_shift).all(), f"the result is wrong with parallel left shift operation"

def test_multi_block_bitwise_op():
    org_marker1 = bytearray([0b11001100, 0b10101010, 0b11001100, 0b10101010])
    org_marker2 = bytearray([0b11110000, 0b00001111, 0b11110000, 0b00001111])
    block_size = 2
    marker1 = np.array(org_marker1)
    marker2 = np.array(org_marker2)
    expected_value_and = np.array([0b11000000, 0b00001010, 0b11000000, 0b00001010])
    print(f"marker dtype: {marker1.dtype}")
    marker = multi_block_parallel_bitwise_and(marker1, marker2, block_size)
    assert (marker == expected_value_and).all(), f"the result is wrong with parallel and operation"

    expected_value_or = np.array([0b11111100, 0b10101111, 0b11111100, 0b10101111])
    marker = multi_block_parallel_bitwise_or(marker1, marker2, block_size)
    assert (marker == expected_value_or).all(), f"the result is wrong with parallel or operation"

    expected_value_left_shift = np.array([0b10011000, 0b01010101, 0b10011000, 0b01010101])
    marker = multi_block_parallel_bitwise_left_shift(marker1, block_size)
    print(marker)
    # print(marker)
    assert (marker == expected_value_left_shift).all(), f"the result is wrong with parallel left shift operation"

    carry_bit_len = 1
    marker_list = np.array([marker1, marker2])
    expected_value_extract = np.array([[0b10101010, 0b11001100],
                                       [0b00001111, 0b11110000]])
    # print(expected_value_extract)
    extract_marker_list, new_block_size = extract_recalculation_data(marker_list, block_size, carry_bit_len)
    assert (expected_value_extract == extract_marker_list).all(), f"the result is wrong with parallel extract operation"

    marker1_correct = extract_marker_list[0]
    marker_correct = multi_block_parallel_bitwise_left_shift(marker1_correct, new_block_size)

    expected_value_correct = np.array([0b10011000, 0b01010101, 0b10011001, 0b01010101])
    marker = correct_result(marker, marker_correct, block_size, new_block_size)
    print(marker)
    assert (expected_value_correct == marker).all(), f"the result is wrong with paraller CORRECT operation"


if __name__ == "__main__":
    test_multi_block_bitwise_op()