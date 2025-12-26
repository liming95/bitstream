#banana
from speculative_carry_bit import *
from utils import *
def banana_rgx_match(b_marker, a_marker, n_marker, block_size):
    shift_num = 0
    b_marker_shifted = multi_block_parallel_bitwise_left_shift(b_marker, block_size)
    print(f'b_shifted:{b_marker_shifted}')
    shift_num += 1
    ba_marker = multi_block_parallel_bitwise_and(b_marker_shifted, a_marker, block_size)
    ba_marker_shifted = multi_block_parallel_bitwise_left_shift(ba_marker, block_size)
    print(f'ba_shifted:{ba_marker_shifted}')
    shift_num += 1
    ban_marker = multi_block_parallel_bitwise_and(ba_marker_shifted, n_marker, block_size)
    ban_marker_shifted = multi_block_parallel_bitwise_left_shift(ban_marker, block_size)
    print(f'ban_shifted:{ban_marker_shifted}')
    shift_num += 1
    bana_marker = multi_block_parallel_bitwise_and(ban_marker_shifted, a_marker, block_size)
    bana_marker_shifted = multi_block_parallel_bitwise_left_shift(bana_marker, block_size)
    print(f'bana_shifted:{bana_marker_shifted}')
    shift_num += 1
    banan_marker = multi_block_parallel_bitwise_and(bana_marker_shifted, n_marker, block_size)
    banan_marker_shifted = multi_block_parallel_bitwise_left_shift(banan_marker, block_size)
    print(f'banan_shifted:{banan_marker_shifted}')
    shift_num += 1
    banana_marker = multi_block_parallel_bitwise_and(banan_marker_shifted, a_marker, block_size)
    # banana_marker_shifted = multi_block_parallel_bitwise_left_shift(banana_marker, block_size)
    # shift_num += 1
    print(f'banana:{banana_marker}')
    return banana_marker, shift_num

def banana():
    sample_file = 'sample1.txt'
    length, streams = get_bit_streams(sample_file)

    marker_b_char = get_special_char_bit_streams(streams, char='b')
    marker_a_char = get_special_char_bit_streams(streams, char='a')
    marker_n_char = get_special_char_bit_streams(streams, char='n')

    block_size = 4
    marker_b_char = np.array(marker_b_char)
    marker_a_char = np.array(marker_a_char)
    marker_n_char = np.array(marker_n_char)
    print(f'b:{marker_b_char}')
    print(f'a:{marker_a_char}')
    print(f'n:{marker_n_char}')
    banana_marker, shift_num = banana_rgx_match(marker_b_char, marker_a_char, marker_n_char, block_size)
    print(banana_marker)

    marker_list = np.array([marker_b_char, marker_a_char, marker_n_char])
    extract_marker_list, new_block_size = extract_recalculation_data(marker_list, block_size, shift_num)

    marker_b_char = extract_marker_list[0]
    marker_a_char = extract_marker_list[1]
    marker_n_char = extract_marker_list[2]
    print(f'extract_b:{marker_b_char}')
    print(f'extract_a:{marker_a_char}')
    print(f'extract_n:{marker_n_char}')

    banana_marker_correct, shift_num_correct = banana_rgx_match(marker_b_char, marker_a_char, marker_n_char, new_block_size)
    marker = correct_result(banana_marker, banana_marker_correct, block_size, new_block_size)

    print(marker)

if __name__ == "__main__":
    banana()


