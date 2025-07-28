# concatenation operation for compacted bit streams
from dataclasses import dataclass
from typing import Callable, Tuple, List
from .utils import debug_print

@dataclass
class CarryBit:
    index: int
    value: int

@dataclass
class CompactBitStream:
    marker: bytearray
    indexes: List[int]

def expand_compacted_bit_stream(compact_stream: CompactBitStream, sp_indexes: List[int]) -> bytearray:
    """
    Expand a compacted bit stream to its full bytearray representation.

    Args:
        compact_stream: A CompactBitStream object containing the marker and indexes.
        sp_indexes: A list of indexes corresponding to the compacted bit stream.

    Returns:
        A bytearray representing the expanded bit stream.
    """
    if not isinstance(compact_stream, CompactBitStream):
        raise TypeError("Input must be a CompactBitStream instance.")

    expanded_stream = bytearray()
    offset = 0
    len_stream = len(compact_stream.indexes)
    for index in sp_indexes:
        if offset >= len_stream:
            expanded_stream.append(0)
            continue
        if (compact_stream.indexes[offset] == index):
            expanded_stream.append(compact_stream.marker[offset])
            offset += 1
            debug_print(f"Expanded stream at index {index}: {expanded_stream[-1]:08b}")
        else:
            expanded_stream.append(0)
    assert len(compact_stream.indexes) == offset, "Indexes do not match the length of the compacted marker."
    return expanded_stream

def concat_compacted_bit_streams(sp_marker1: CompactBitStream, sp_marker2: CompactBitStream,
                                 sp_indexes: list,
                                 length: int,
                                 sp_carry_bit: CarryBit
                                 ) -> Tuple[CompactBitStream, CarryBit]:
    """
    Concatenate two compacted bit streams.
    Args:
        marker1: The first compacted bit stream.
        marker2: The second compacted bit stream.
        sp_indexes: The index lists of the compacted bit streams.
        length: The length of the bit streams.
        carry_bit: The carry bit to be used in the concatenation.
    Returns:
        A tuple containing the concatenated compacted bit stream, the updated index list, and the carry bit.
    """
    if not isinstance(sp_marker1, CompactBitStream) or not isinstance(sp_marker2, CompactBitStream):
        raise TypeError("Both inputs must be CompactBitStream instances.")
    if not isinstance(sp_indexes, list):
        raise TypeError("sp_indexes must be a list.")
    if not isinstance(sp_carry_bit, CarryBit):
        raise TypeError("carry_bit must be a CarryBit instance.")

    marker1 = expand_compacted_bit_stream(sp_marker1, sp_indexes)
    marker2 = expand_compacted_bit_stream(sp_marker2, sp_indexes)
    debug_print(f"Concatenating compacted bit streams: marker1={marker1}, marker2={marker2}, length={length}, carry_bit={sp_carry_bit}")
    marker = bytearray()
    marker_index = []
    carry_index = sp_carry_bit.index
    carry_bit = sp_carry_bit.value

    for i in range(0, length):
        if i >= len(sp_indexes):
            raise IndexError("Index out of range for one of the streams.")
        carry_index_tmp = sp_indexes[i]
        carry_bit_tmp = (marker1[i] & 0x80) >> 7
        assert carry_index_tmp > carry_index, f"carry_index_tmp {carry_index_tmp} must be greater than or equal to carry_index {carry_index}."
        carry_bit = carry_bit if (carry_index_tmp - carry_index) == 1 else 0

        marker_shift = (marker1[i] << 1 | carry_bit) & 0xFF
        marker_tmp = (marker_shift & marker2[i]) & 0xFF
        if marker_tmp != 0:
            marker.append(marker_tmp)
            marker_index.append(carry_index_tmp)
        carry_index = carry_index_tmp
        carry_bit = carry_bit_tmp

    sp_carry_bit = CarryBit(index=carry_index, value=carry_bit)

    return CompactBitStream(marker, marker_index), sp_carry_bit

# alternation operation for compacted bit streams
def alt_compacted_bit_streams(*marker, sp_indexes, streams):
    """
    Create an alternation of multiple compacted bit streams.

    Args:
        marker: A marker to be included in the alternation.
        indexes: The index lists of the compacted bit streams.
        streams: The compacted bit streams to alternate.

    Returns:
        A single alternated compacted bit stream.
    """
    return marker

# Kleene Star operation for compacted bit streams
def kleene_star_compacted_bit_streams(marker, sp_indexes, streams):
    """
    Apply Kleene star operation to multiple compacted bit streams.

    Args:
        marker: A marker to be included in the Kleene star operation.
        indexes: The index lists of the compacted bit streams.
        streams: The compacted bit streams to apply the Kleene star operation on.

    Returns:
        A single compacted bit stream representing the Kleene star operation.
    """
    return marker

def test_concat_compacted_bit_streams():
    """
    Test the concatenation of compacted bit streams.
    """
    sp_marker1 = CompactBitStream(bytearray([0b00000000, 0b11001100, 0b10101010, 0b10000000]), [0, 1, 2, 3])
    sp_marker2 = CompactBitStream(bytearray([0b00001111, 0b11110000]), [2, 3])
    sp_indexes = [0, 1, 2, 3]
    sp_carry_bit = CarryBit(index=-1, value=0)

    result, carry_bit = concat_compacted_bit_streams(sp_marker1, sp_marker2, sp_indexes, 4, sp_carry_bit)

    assert result.marker == bytearray([0b00000101])
    assert result.indexes == [2]
    assert carry_bit.index == 3
    assert carry_bit.value == 1
    print("Test passed!")

if __name__ == "__main__":
    test_concat_compacted_bit_streams()
    print("All passed!")