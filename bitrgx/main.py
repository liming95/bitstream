#!/usr/bin/python3

from bitregex import bitrgx, utils, sp_bitrgx
"""
    main.py: Test the regex expression (ba(na)*) | (apple) with the provided streams.
"""

def match_regex_expression(streams, carry_bits_dict, length=1):
    """
    Match a regex expression against the provided streams.

    Args:
        streams: A list of bytearrays representing the bit streams.
        carry_bits: A dictionary to hold carry bits.

    Returns:
        A tuple containing the marker and updated carry bits.
    """
    marker = bytearray()
    # get the special character bit streams
    marker_b_char = utils.get_special_char_bit_streams(streams, char='b')
    marker_a_char = utils.get_special_char_bit_streams(streams, char='a')
    marker_n_char = utils.get_special_char_bit_streams(streams, char='n')
    marker_p_char = utils.get_special_char_bit_streams(streams, char='p')
    marker_l_char = utils.get_special_char_bit_streams(streams, char='l')
    marker_e_char = utils.get_special_char_bit_streams(streams, char='e')

    # match the regex expression (ba(na)*)
    # match 'ba'
    carry_bit = carry_bits_dict['ba']
    marker_ba, carry_bit = bitrgx.concat_bit_streams(marker_b_char, marker_a_char, length=length, carry_bit=carry_bit)
    carry_bits_dict['ba'] = carry_bit

    def matcher_na (marker_pre, markers_other, length, carry_bit_dict_na):
        carry_bit = carry_bit_dict_na['-n']
        marker_n, carry_bit = bitrgx.concat_bit_streams(marker_pre, marker_n_char, length, carry_bit)
        carry_bit_dict_na['-n'] = carry_bit
        carry_bit = carry_bit_dict_na['na']
        marker_na, carry_bit = bitrgx.concat_bit_streams(marker_n, marker_a_char, length, carry_bit)
        carry_bit_dict_na['na'] = carry_bit
        carry_bit_dict_na['bit_len_used'] = 1
        return marker_na, carry_bit_dict_na

    # match 'ba(na)*' with Kleene star
    carry_bits_dict_sub = carry_bits_dict['ba(na)*']
    marker_ba_naks, carry_bits_dict_sub = bitrgx.kleene_star_bit_streams(marker_ba, matcher_na, length=length, carry_bits=carry_bits_dict_sub)
    carry_bits_dict['ba(na)*'] = carry_bits_dict_sub

    # match the regex expression 'apple'
    #match 'ap'
    carry_bit = carry_bits_dict['ap']
    marker_ap, carry_bit = bitrgx.concat_bit_streams(marker_a_char, marker_p_char, length, carry_bit)
    carry_bits_dict['ap'] = carry_bit
    #match 'app'
    carry_bit = carry_bits_dict['app']
    marker_app, carry_bit = bitrgx.concat_bit_streams(marker_ap, marker_p_char, length, carry_bit)
    carry_bits_dict['app'] = carry_bit
    #match 'appl'
    carry_bit = carry_bits_dict['appl']
    marker_appl, carry_bit = bitrgx.concat_bit_streams(marker_app, marker_l_char, length, carry_bit)
    carry_bits_dict['appl'] = carry_bit
    #match 'apple'
    carry_bit = carry_bits_dict['apple']
    marker_apple, carry_bit = bitrgx.concat_bit_streams(marker_appl, marker_e_char, length, carry_bit)
    carry_bits_dict['apple'] = carry_bit

    # match the regex expression 'ba(na)* | apple'
    marker = bitrgx.alt_bit_streams(marker_ba_naks, marker_apple, length)

    return marker, carry_bits_dict

def test_regex_expression1(streams: list[bytearray], length):
    byte_num = length // 8
    # pad the streams to be a multiple of 4 bytes
    if byte_num % 4 != 0:
        for stream in streams:
            stream.extend([0] * (4 - (byte_num % 4)))

    carry_bits_dict = {
        'bit_len_used': 0,
        'ba': int(0),
        'ba(na)*': {'bit_len_used': 0, '-n':bytearray([0]),'na': bytearray([0])},
        'ap': int(0),
        'app': int(0),
        'appl': int(0),
        'apple': int(0),
    }
    marker = bytearray()
    for i in range(0, byte_num, 4):
        stream_part = [stream[i:i + 4] for stream in streams]
        marker_tmp, carry_bits_dict = match_regex_expression(stream_part, carry_bits_dict, length = 4)
        marker.extend(marker_tmp)

    print('Marker:' + ' '.join(f'0x{b:02X}' for b in marker))
    return marker, carry_bits_dict

def match_regex_expression_sp(compacted_marker_dict, sp_indexes, carry_bits_dict, length):
    # get the compacted marker within bounds of sp_indexes
    start = sp_indexes[0]
    end = sp_indexes[-1]
    compacted_marker_b = compacted_marker_dict['b'].marker_within_bounds(start, end)
    compacted_marker_a = compacted_marker_dict['a'].marker_within_bounds(start, end)
    compacted_marker_n = compacted_marker_dict['n'].marker_within_bounds(start, end)
    compacted_marker_p = compacted_marker_dict['p'].marker_within_bounds(start, end)
    compacted_marker_l = compacted_marker_dict['l'].marker_within_bounds(start, end)
    compacted_marker_e = compacted_marker_dict['e'].marker_within_bounds(start, end)

    # match the regex expression (ba(na)*)
    # match 'ba'
    carry_bit = carry_bits_dict['ba']
    marker_ba, carry_bit = sp_bitrgx.concat_compacted_bit_streams(compacted_marker_b, compacted_marker_a, sp_indexes, length=length, sp_carry_bit=carry_bit)
    carry_bits_dict['ba'] = carry_bit

    def matcher_na (marker_pre, markers_other, sp_indexes, length, carry_bit_dict_na):
        carry_bit = carry_bit_dict_na['-n']
        marker_n, carry_bit = sp_bitrgx.concat_compacted_bit_streams(marker_pre, compacted_marker_n, sp_indexes, length, carry_bit)
        carry_bit_dict_na['-n'] = carry_bit
        carry_bit = carry_bit_dict_na['na']
        marker_na, carry_bit = sp_bitrgx.concat_compacted_bit_streams(marker_n, compacted_marker_a, sp_indexes, length, carry_bit)
        carry_bit_dict_na['na'] = carry_bit
        carry_bit_dict_na['bit_len_used'] = 1
        return marker_na, carry_bit_dict_na

    # match 'ba(na)*' with Kleene star
    carry_bits_dict_sub = carry_bits_dict['ba(na)*']
    marker_ba_naks, carry_bits_dict_sub = sp_bitrgx.kleene_star_compacted_bit_streams(marker_ba, sp_indexes, matcher_na, length=length, carry_bits_dict=carry_bits_dict_sub)
    carry_bits_dict['ba(na)*'] = carry_bits_dict_sub

    # match the regex expression 'apple'
    #match 'ap'
    carry_bit = carry_bits_dict['ap']
    marker_ap, carry_bit = sp_bitrgx.concat_compacted_bit_streams(compacted_marker_a, compacted_marker_p, sp_indexes, length, carry_bit)
    carry_bits_dict['ap'] = carry_bit
    #match 'app'
    carry_bit = carry_bits_dict['app']
    marker_app, carry_bit = sp_bitrgx.concat_compacted_bit_streams(marker_ap, compacted_marker_p, sp_indexes, length, carry_bit)
    carry_bits_dict['app'] = carry_bit
    #match 'appl'
    carry_bit = carry_bits_dict['appl']
    marker_appl, carry_bit = sp_bitrgx.concat_compacted_bit_streams(marker_app, compacted_marker_l, sp_indexes, length, carry_bit)
    carry_bits_dict['appl'] = carry_bit
    #match 'apple'
    carry_bit = carry_bits_dict['apple']
    marker_apple, carry_bit = sp_bitrgx.concat_compacted_bit_streams(marker_appl, compacted_marker_e, sp_indexes, length, carry_bit)
    carry_bits_dict['apple'] = carry_bit

    # match the regex expression 'ba(na)* | apple'
    marker = sp_bitrgx.alt_compacted_bit_streams(marker_ba_naks, marker_apple, sp_indexes, length)

    return marker, carry_bits_dict


def test_regex_expression2(streams, length):
    # get the special character bit streams
    marker_b_char = utils.get_special_char_bit_streams(streams, char='b')
    marker_a_char = utils.get_special_char_bit_streams(streams, char='a')
    marker_n_char = utils.get_special_char_bit_streams(streams, char='n')
    marker_p_char = utils.get_special_char_bit_streams(streams, char='p')
    marker_l_char = utils.get_special_char_bit_streams(streams, char='l')
    marker_e_char = utils.get_special_char_bit_streams(streams, char='e')

    indexes_sp_b, marker_sp_b = utils.compact_sparse_marker_to_index_list(marker_b_char)
    indexes_sp_a, marker_sp_a = utils.compact_sparse_marker_to_index_list(marker_a_char)
    indexes_sp_n, marker_sp_n = utils.compact_sparse_marker_to_index_list(marker_n_char)
    indexes_sp_p, marker_sp_p = utils.compact_sparse_marker_to_index_list(marker_p_char)
    indexes_sp_l, marker_sp_l = utils.compact_sparse_marker_to_index_list(marker_l_char)
    indexes_sp_e, marker_sp_e = utils.compact_sparse_marker_to_index_list(marker_e_char)

    indexes_lists = [indexes_sp_b, indexes_sp_a, indexes_sp_n, indexes_sp_p, indexes_sp_l, indexes_sp_e]
    indexes_sp = utils.merge_and_sort_index_lists(indexes_lists)
    utils.debug_print(f"indexes_sp: {indexes_sp}")
    compacted_marker_dict = {'b': sp_bitrgx.CompactBitStream(marker_sp_b, indexes_sp_b),
                             'a': sp_bitrgx.CompactBitStream(marker_sp_a, indexes_sp_a),
                             'n': sp_bitrgx.CompactBitStream(marker_sp_n, indexes_sp_n),
                             'p': sp_bitrgx.CompactBitStream(marker_sp_p, indexes_sp_p),
                             'l': sp_bitrgx.CompactBitStream(marker_sp_l, indexes_sp_l),
                             'e': sp_bitrgx.CompactBitStream(marker_sp_e, indexes_sp_e),
    }

    carry_bits_dict = {
        'bit_len_used': sp_bitrgx.CarryBit(-1, 0),
        'ba': sp_bitrgx.CarryBit(-1, 0),
        'ba(na)*': {'bit_len_used': 0, '-n':[],'na': []},
        'ap': sp_bitrgx.CarryBit(-1, 0),
        'app': sp_bitrgx.CarryBit(-1, 0),
        'appl': sp_bitrgx.CarryBit(-1, 0),
        'apple': sp_bitrgx.CarryBit(-1, 0),
    }
    compacted_marker = sp_bitrgx.CompactBitStream()
    rmd = length % 4
    for i in range(0, len(indexes_sp), 4):
        indexes_part = indexes_sp[i:i+4] if i+4 <= len(indexes_sp) else indexes_sp[i:i+rmd]
        compacted_marker_tmp, carry_bits_dict = match_regex_expression_sp(compacted_marker_dict, indexes_part, carry_bits_dict, 4)
        compacted_marker.extend(compacted_marker_tmp)
    print(f'Indexes: {compacted_marker.indexes} Marker:' + ' '.join(f'0x{b:02X}' for b in compacted_marker.marker))
    #print('Marker:' + ' '.join(f'0x{b:02X}' for b in marker))
    return compacted_marker, carry_bits_dict

def main():
    """
    Main function to run the bitrgx tests.
    regex expression (ba(na)*) | (apple)
    """
    print("Running bitrgx tests...")
    # get bit streams from file
    sample_file = 'sample.txt'
    length, streams = utils.get_bit_streams(sample_file)
    utils.debug_print(f"Bit streams length: {length} Bit streams: {streams}")

    marker, carry_bits_dict = test_regex_expression1(streams, length)

    compacted_marker, carry_bits_dict_sp = test_regex_expression2(streams, length)


    print("All tests passed!")

if __name__ == "__main__":
    main()