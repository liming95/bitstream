#1. character -> bit stream
def char_to_bit_stream(data: bytearray):
    """
    Convert 8 characters to a bit stream.

    Args:
        data: The characters to convert.

    Returns:
        A bit stream representing these characters.
    example:
        >>> char_to_bit_stream(bytearray('12345678'))
        returns : bytearray(b'Ufx\x80\xff\xff\x00\x00')
    """
    if not isinstance(data, bytearray):
        raise TypeError("Input must be a bytearray.")
    if len(data) != 8:
        raise ValueError("Input must be exactly 8 bytes long.")
    bit_stream = bytearray(b'\x00' * 8)
    pos = 0
    for byte in data:
        for i in range(8):
            # Convert each byte to its binary representation
            # and append it to the bit stream
            bitmask = 1 << i
            bit_stream[i] |= (byte & bitmask) >> i << pos
        pos += 1
    return bit_stream

def get_bit_streams(filename: str):
    """
    Read a file and return its content as a bit stream.

    Args:
        filename: The name of the file to read.

    Returns:
        A bytearray representing the bit stream of the file content.
    """
    with open(filename, 'rb') as f:
        data = f.read()
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("File content must be bytes or bytearray.")
    length = len(data)
    streams = []
    for i in range(0, 8):
        streams.append(bytearray())

    if length % 8 != 0:
        data += b'\x00' * (8 - length % 8)  # Pad to multiple of 8 bytes
    for i in range(0, len(data), 8):
        if i + 8 > len(data):
            raise ValueError("File content must be a multiple of 8 bytes.")
        bit_stream = char_to_bit_stream(bytearray(data[i:i + 8]))
        for j in range(8):
            streams[j] += bit_stream[j:j + 1]
    return length, streams

#2. bit stream -> bit stream for special characters (marker, etc.)
def bitwise_xnor(a: int, b: int, length: int):
    """
    Perform bitwise XNOR operation on two integers.

    Args:
        a: The first integer.
        b: The second integer.
        length: The number of bits to consider.

    Returns:
        The result of the XNOR operation as an integer.
    """
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("Inputs must be integers.")
    if length <= 0 or length > 32:
        raise ValueError("Length must be between 1 and 32.")

    mask = (1 << length) - 1
    return ~(a ^ b) & mask


def get_special_char_bit_streams(streams, char=None):
    """
    Convert a list of bit streams to a  bit stream for special character.

    Args:
        streams: A list of bit streams.

    Returns:
        A bit stream.
    """
    if not isinstance(streams, list) or not all(isinstance(s, bytearray) for s in streams):
        raise TypeError("Input must be a list of bytearrays.")

    # Example implementation: simply concatenate the streams
    bitmask = bytearray(b'\x00' * 8)
    if char is not None:
        if not isinstance(char, str) or len(char) != 1:
            raise ValueError("Character must be a single character string.")
    else:
        print("No special character provided, using default marker.")
    for i in range(0, 8):
        if ord(char[0]) & (1 << i):
            bitmask[i] = 0xFF
        else:
            bitmask[i] = 0x00

    assert len(streams) == 8, "There must be exactly 8 streams."
    marker = bytearray()
    for i in range(0,len(streams[0])):
        bit_stream = 0xFF
        for j in range(0, len(streams)):
            bit_stream &= bitwise_xnor(streams[j][i], bitmask[j], 8)
        marker.append(bit_stream)
    return marker

#3. compact sparse marker to index list and bit streams
def compact_sparse_marker_to_index_list(marker: bytearray):
    """
    Convert a bytearray of marker to a compacted index list.

    Args:
        marker: A bytearray of marker.

    Returns:
        A list of indexes where the marker are set.
    """
    if not isinstance(marker, bytearray):
        raise TypeError("Input must be a bytearray.")

    indexes = []
    compacted_marker = bytearray()
    for i in range(len(marker)):
        if marker[i] != 0:
            indexes.append(i)
            compacted_marker.append(marker[i])

    return indexes, compacted_marker

#4. merge and sort index lists
def merge_and_sort_index_lists(*index_lists):
    """
    Merge and sort multiple index lists.

    Args:
        index_lists: A variable number of index lists to merge and sort.

    Returns:
        A single sorted list of indexes.
    """
    if not all(isinstance(lst, list) for lst in index_lists):
        raise TypeError("All inputs must be lists.")

    merged_list = []
    for lst in index_lists:
        merged_list.extend(lst)

    return sorted(set(merged_list))  # Remove duplicates and sort

def test_char_to_bit_stream():
    """
    Test the char_to_bit_stream function with a sample input.
    """
    sample_input = bytearray('12345678',encoding='ascii')
    expected_output = bytearray(b'Ufx\x80\xff\xff\x00\x00')
    result = char_to_bit_stream(sample_input)
    assert result == expected_output, f"Expected {expected_output}, got {result}"
    print("Test passed!")

def test_get_bit_streams():
    """
    Test the get_bit_streams function with a sample file.
    """
    sample_file = 'sample.txt'
    with open(sample_file, 'wb') as f:
        f.write(bytearray('1234567812345678', encoding='ascii'))  # Create a sample file with 16 bytes

    length, streams = get_bit_streams(sample_file)
    expected_length = 16
    expected_streams = [bytearray(b'UU'), bytearray(b'ff'), bytearray(b'xx'),
                        bytearray(b'\x80\x80'), bytearray(b'\xff\xff'),
                        bytearray(b'\xff\xff'), bytearray(b'\x00\x00'),
                        bytearray(b'\x00\x00')]

    assert length == expected_length, f"Expected {expected_length}, got {length}"
    for i in range(8):
        assert streams[i] == expected_streams[i], f"Stream {i} mismatch: expected {expected_streams[i]}, got {streams[i]}"

    print("Test passed!")

def test_get_special_char_bit_streams():
    """
    Test the get_special_char_bit_streams function with a sample input.
    """
    streams = [bytearray(b'UU'), bytearray(b'ff'), bytearray(b'xx'),
                        bytearray(b'\x80\x80'), bytearray(b'\xff\xff'),
                        bytearray(b'\xff\xff'), bytearray(b'\x00\x00'),
                        bytearray(b'\x00\x00')]
    char = '1'
    result = get_special_char_bit_streams(streams, char)
    expected_result = bytearray(b'\x01\x01')  # Example expected result
    assert result == expected_result, f"Expected {expected_result}, got {result}"
    print("Test passed!")

def test_compact_sparse_marker_to_index_list():
    """
    Test the compact_sparse_marker_to_index_list function with a sample input.
    """
    marker = bytearray(b'\x01\x00\x01\x00\x01\x00\x01\x00')
    expected_indexes = [0, 2, 4, 6]
    expected_compacted_marker = bytearray(b'\x01\x01\x01\x01')

    indexes, compacted_marker = compact_sparse_marker_to_index_list(marker)

    assert indexes == expected_indexes, f"Expected {expected_indexes}, got {indexes}"
    assert compacted_marker == expected_compacted_marker, f"Expected {expected_compacted_marker}, got {compacted_marker}"

    print("Test passed!")

def test_merge_and_sort_index_lists():
    """
    Test the merge_and_sort_index_lists function with sample inputs.
    """
    list1 = [1, 3, 5]
    list2 = [2, 4, 6]
    list3 = [0, 7, 8]

    expected_result = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    result = merge_and_sort_index_lists(list1, list2, list3)

    assert result == expected_result, f"Expected {expected_result}, got {result}"

    print("Test passed!")

if __name__ == "__main__":
    test_char_to_bit_stream()
    test_get_bit_streams()
    test_get_special_char_bit_streams()
    test_compact_sparse_marker_to_index_list()
    test_merge_and_sort_index_lists()
    # import sys
    # if len(sys.argv) != 2:
    #     print("Usage: python utils.py <bytearray>")
    #     sys.exit(1)
    # input_data = bytearray(sys.argv[1], encoding='ascii')
    # if len(input_data) != 8:
    #     print("Error: Input must be exactly 8 bytes long.")
    #     sys.exit(1)
    # try:
    #     result = char_to_bit_stream(input_data)
    #     print("Bit stream:", result)
    # except (TypeError, ValueError) as e:
    #     print("Error:", e)