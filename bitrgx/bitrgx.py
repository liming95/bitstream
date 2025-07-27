# Concatenation operations for bit streams
def concat_bit_streams(marker1: bytearray, marker2: bytearray, length: int = 1, carry_bit: int = 0):
    """
    Concatenate two bit streams into a single stream.

    Args:
        marker1: The first bit stream to concatenate.
        marker2: The second bit stream to concatenate.

    Returns:
        A single concatenated bit stream.
    """
    if not isinstance(marker1, bytearray) or not isinstance(marker2, bytearray):
        raise TypeError("Both inputs must be bytearrays.")
    assert len(marker1) == len(marker2), f"Both streams must have the same length.{len(marker1)} != {len(marker2)}"
    marker = bytearray()
    carry_bit_tmp = 0
    for i in range(0, length):
        if i >= len(marker1) or i >= len(marker2):
            raise IndexError("Index out of range for one of the streams.")
        carry_bit_tmp = (marker1[i] & 0x80) >> 7
        marker_shift = (marker1[i] << 1 | carry_bit) & 0xFF
        carry_bit = carry_bit_tmp
        marker.append(marker_shift & marker2[i])
    return marker, carry_bit

# Alternation operations for bit streams
def alt_bit_streams(marker1: bytearray, marker2: bytearray, length: int = 1):
    """
    Create an alternation of two bit streams.

    Args:
        marker1: The first bit stream to alternate.
        marker2: The second bit stream to alternate.

    Returns:
        A single alternated bit stream.
    """
    if not isinstance(marker1, bytearray) or not isinstance(marker2, bytearray):
        raise TypeError("Both inputs must be bytearrays.")
    assert len(marker1) == len(marker2), f"Both streams must have the same length.{len(marker1)} != {len(marker2)}"
    marker = bytearray()
    for i in range(0, length):
        if i >= len(marker1) or i >= len(marker2):
            raise IndexError("Index out of range for one of the streams.")
        marker.append(marker1[i] | marker2[i])

    return marker

#Kleene Star operations for bit streams
from typing import Callable, Tuple
def get_dict_spc_bit(carry_bits: dict, pos: int) -> dict:
    """
    Get the special bit from the carry bits dictionary.

    Args:
        carry_bits: A dictionary containing carry bits.
        pos: The position to retrieve the special bit from.

    Returns:
        The special bit at the specified position.
    """
    carry_bit_spc = {}
    for key, value in carry_bits.items():
        index = pos // 8
        offset = pos % 8
        spc_bit = value[index] & (1 << offset)
        carry_bit_spc[key] = spc_bit >> offset

    return carry_bit_spc

def update_dict_spc_bit(carry_bits: dict, carry_bit: dict, pos: int):
    """
    Update the carry bits dictionary with a new special bit.

    Args:
        carry_bits: A dictionary containing carry bits.
        carry_bit: The new special bit to update.
        pos: The position to update the special bit at.

    Returns:
        None
    """
    assert isinstance(carry_bits, dict), "carry_bits must be a dictionary."
    assert isinstance(carry_bit, dict), "carry_bit must be a dictionary."
    for key, value in carry_bit.items():
        index = pos // 8
        offset = pos % 8
        assert key in carry_bits and isinstance(carry_bits[key], bytearray), f"Key {key} must exist in carry_bits and be a bytearray."

        if index >= len(carry_bits[key]):
            carry_bits[key].extend([0] * (index - len(carry_bits[key]) + 1))
        carry_bits[key][index] |= (value << offset)

def kleene_star_bit_streams(
    marker1: bytearray,
    matcher: Callable[[bytearray, list, int, dict], Tuple[bytearray, dict]],
    length: int = 1,
    carry_bits: dict = None
    ) -> Tuple[bytearray, int]:
    """
    Apply Kleene star operation to a bit stream.
    """
    if not isinstance(marker1, bytearray):
        raise TypeError("Input must be a bytearray.")
    if not callable(matcher):
        raise TypeError("Matcher must be a callable function.")

    marker = bytearray(marker1)
    carry_bits_out = {}
    marker_tmp = bytearray(marker1)
    carry_bit_tmp = {}

    iterations = 0
    while any(marker_tmp) or (iterations < carry_bits.get('ks_shift_num', 0)):  # Limit iterations to prevent infinite loop
        carry_bit_tmp = get_dict_spc_bit(carry_bits, iterations)
        marker_tmp, carry_bit_tmp = matcher(marker1, [], length, carry_bit_tmp)
        marker |= bytearray([a | b for a, b in zip(marker, marker_tmp)])
        update_dict_spc_bit(carry_bits_out, carry_bit_tmp, iterations)
        iterations += 1
    carry_bits_out['ks_shift_num'] = iterations

    return marker, carry_bits_out

def test_concat_bit_streams():
    """
    Test the concat_bit_streams function.
    """
    marker1 = bytearray([0b11001100, 0b10101010])
    marker2 = bytearray([0b11110000, 0b00001111])
    expect_marker = bytearray([0b10010000, 0b00000101])
    expect_carry_bit = 1
    length = 2
    carry_bit = 0
    result, carry_bit_out = concat_bit_streams(marker1, marker2, length, carry_bit)
    assert result == expect_marker, f"Expected {expect_marker}, got {result}"
    assert carry_bit_out == expect_carry_bit, f"Expected carry bit {expect_carry_bit}, got {carry_bit_out}"
    print("Test passed!")

def test_alt_bit_streams():
    """
    Test the alt_bit_streams function.
    """
    marker1 = bytearray([0b11001100, 0b10101010])
    marker2 = bytearray([0b11110000, 0b00001111])
    expect_marker = bytearray([0b11111100, 0b10101111])
    length = 2
    result = alt_bit_streams(marker1, marker2, length)
    assert result == expect_marker, f"Expected {expect_marker}, got {result}"
    print("Test passed!")

def test_kleene_star_bit_streams():
    """
    Test the kleene_star_bit_streams function.
    """
    marker1 = bytearray([0b11001100, 0b10101010])
    def matcher(marker, indexes, length, carry_bits):
        # Simple matcher that returns the same marker and carry bits

        return marker, carry_bits

    expect_marker = bytearray([0b11001100, 0b10101010])
    expect_carry_bits = {'ks_shift_num': 1}
    result_marker, result_carry_bits = kleene_star_bit_streams(marker1, matcher)

    assert result_marker == expect_marker, f"Expected {expect_marker}, got {result_marker}"
    assert result_carry_bits == expect_carry_bits, f"Expected {expect_carry_bits}, got {result_carry_bits}"
    print("Test passed!")

if __name__ == "__main__":
    test_concat_bit_streams()
    test_alt_bit_streams()
    print("All tests passed!")