# concatenation operation for compacted bit streams
from dataclasses import dataclass
from typing import Callable, Tuple, List
from .utils import debug_print

@dataclass
class CarryBit:
    index: int
    value: int
@dataclass
class CarryBits:
    carry_bit: list[CarryBit]

class CompactBitSet:
    def __init__(self, marker: bytearray = None, indexes: List[int] = None):
        self.marker = marker if marker is not None else bytearray()
        self.indexes = indexes if indexes is not None else []

    def copy(self, other: 'CompactBitSet'):
        self.marker = other.marker.copy()
        self.indexes = other.indexes.copy()

    def update(self, other: 'CompactBitSet'):
        for i in range(len(other.indexes)):
            if other.indexes[i] not in self.indexes:
                # Append new index and marker by sorted order
                if other.indexes[i] > self.indexes[-1]:
                    self.indexes.append(other.indexes[i])
                    self.marker.append(other.marker[i])
                    continue
                for j in range(len(self.indexes)):
                    if other.indexes[i] < self.indexes[j]:
                        self.indexes.insert(j, other.indexes[i])
                        self.marker.insert(j, other.marker[i])
                        break
            else:
                index = self.indexes.index(other.indexes[i])
                self.marker[index] |= other.marker[i]

    def add(self, spc_marker: 'int', index: 'int'):
        self.marker.append(spc_marker)
        self.indexes.append(index)

    def extend(self, other: 'CompactBitSet'):
        self.marker.extend(other.marker)
        self.indexes.extend(other.indexes)

    def marker_within_bounds(self, start, end):
        compacted_stream = CompactBitSet()
        indexes = self.indexes
        marker = self.marker

        for i in range(len(indexes)):
            if not (start <= indexes[i] <= end):
                continue
            compacted_stream.add(marker[i], indexes[i])
        return compacted_stream


def expand_compacted_set_bitmap(compact_stream: CompactBitSet, sp_indexes: List[int]) -> bytearray:
    """
    Expand a compacted bit stream to its full bytearray representation.

    Args:
        compact_stream: A CompactBitSet object containing the marker and indexes.
        sp_indexes: A list of indexes corresponding to the compacted bit stream.

    Returns:
        A bytearray representing the expanded bit stream.
    """
    if not isinstance(compact_stream, CompactBitSet):
        raise TypeError("Input must be a CompactBitSet instance.")

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

def concat_compacted_set_bitmaps(sp_marker1: CompactBitSet, sp_marker2: CompactBitSet,
                                 sp_indexes: list,
                                 length: int,
                                 sp_carry_bit: CarryBit
                                 ) -> Tuple[CompactBitSet, CarryBit]:
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
    if not isinstance(sp_marker1, CompactBitSet) or not isinstance(sp_marker2, CompactBitSet):
        raise TypeError("Both inputs must be CompactBitSet instances.")
    if not isinstance(sp_indexes, list):
        raise TypeError("sp_indexes must be a list.")
    if not isinstance(sp_carry_bit, CarryBit):
        raise TypeError("carry_bit must be a CarryBit instance.")

    marker1 = expand_compacted_set_bitmap(sp_marker1, sp_indexes)
    marker2 = expand_compacted_set_bitmap(sp_marker2, sp_indexes)
    debug_print(f"Concatenating compacted bit streams: marker1={marker1}, marker2={marker2}, length={length}, carry_bit={sp_carry_bit}")
    marker = bytearray()
    marker_index = []
    marker_set = bytearray()
    marker_set_index = []

    carry_index = sp_carry_bit.index
    carry_bit = sp_carry_bit.value

    length = min(length, len(sp_indexes))
    for i in range(0, length):
        if i >= len(sp_indexes):
            raise IndexError("Index out of range for one of the streams.")
        marker_set_tmp = marker1[i] & marker2[i]
        carry_index_tmp = sp_indexes[i]
        carry_bit_tmp = (marker1[i] & 0x80) >> 7
        assert carry_index_tmp > carry_index, f"carry_index_tmp {carry_index_tmp} must be greater than or equal to carry_index {carry_index}."
        carry_bit = carry_bit if (carry_index_tmp - carry_index) == 1 else 0

        marker_shift = (marker1[i] << 1 | carry_bit) & 0xFF
        marker_tmp = (marker_shift & marker2[i]) & 0xFF
        marker_set_tmp |= marker_tmp

        debug_print(f"Concatenated marker at index {i}: {marker_tmp:08b}, carry_index={carry_index_tmp}, carry_bit={carry_bit}")
        if marker_tmp != 0:
            debug_print(f"Appending marker_tmp: {marker_tmp:08b} at index {carry_index_tmp}")
            marker.append(marker_tmp)
            debug_print(f"Current marker: {marker}")
            marker_index.append(carry_index_tmp)
        carry_index = carry_index_tmp
        carry_bit = carry_bit_tmp
        if marker_set_tmp != 0:
            debug_print(f"Appending marker_set: {marker_set_tmp:08b} at index {carry_index_tmp}")
            marker_set.append(marker_set_tmp)
            debug_print(f"Current marker after set: {marker_set}")
            marker_set_index.append(carry_index_tmp)

    sp_carry_bit = CarryBit(index=carry_index, value=carry_bit)
    debug_print(f"Final concatenated marker: {marker}, carry_bit={sp_carry_bit}")
    return CompactBitSet(marker_set, marker_set_index), CompactBitSet(marker, marker_index), sp_carry_bit

# alternation operation for compacted bit streams
def alt_compacted_set_bitmaps(sp_marker1: CompactBitSet, sp_marker2: CompactBitSet,
                              sp_indexes: list,
                              length: int
                              ) -> CompactBitSet:
    """
    Create an alternation of two compacted bit streams.
    Args:
        sp_marker1: The first compacted bit stream.
        sp_marker2: The second compacted bit stream.
        indexes: The index lists of the compacted bit streams.
        length: The length of the bit streams.
    Returns:
        A single compacted bit stream representing the alternation.
    """
    if not isinstance(sp_marker1, CompactBitSet) or not isinstance(sp_marker2, CompactBitSet):
        raise TypeError("Both inputs must be CompactBitSet instances.")
    if not isinstance(sp_indexes, list):
        raise TypeError("sp_indexes must be a list.")
    marker1 = expand_compacted_set_bitmap(sp_marker1, sp_indexes)
    marker2 = expand_compacted_set_bitmap(sp_marker2, sp_indexes)
    debug_print(f"Alternating compacted bit streams: marker1={marker1}, marker2={marker2}, length={length}")
    marker = bytearray()
    marker_index = []

    length = min(length, len(sp_indexes))
    for i in range(0, length):
        if i >= len(sp_indexes):
            raise IndexError("Index out of range for one of the streams.")
        marker_tmp = marker1[i] | marker2[i]
        if marker_tmp != 0:
            marker.append(marker_tmp)
            marker_index.append(sp_indexes[i])
    return CompactBitSet(marker, marker_index)

# Kleene Star operation for compacted bit streams
def get_dict_spc_pos(carry_bits_dict: dict, pos: int) -> dict:
    """
    Get the special bit from the carry bits dictionary.

    Args:
        carry_bits: A dictionary containing carry bits.
        pos: The position to retrieve the special bit from.

    Returns:
        A dictionary containing the special bit for the given position.
    """
    if not isinstance(carry_bits_dict, dict):
        raise TypeError("carry_bits must be a dictionary.")
    if not isinstance(pos, int):
        raise TypeError("pos must be an integer.")

    bit_len_used = carry_bits_dict.get('bit_len_used', 0)
    carry_bit_dict = {}
    for key, value in carry_bits_dict.items():
        if key == 'bit_len_used':
            continue
        assert isinstance(value, list), f"Value for key {key} must be a list."
        if pos >= len(value):
            carry_bit_dict[key] = CarryBit(-1, 0)
        else:
            carry_bit_dict[key] = value[pos]
    carry_bit_dict['bit_len_used'] = 1
    return carry_bit_dict

def update_dict_spc_pos(carry_bits_dict: dict, carry_bit_dict: dict, pos: int):
    """
    Update the carry bits dictionary with a new special bit.

    Args:
        carry_bits: A dictionary containing carry bits.
        carry_bit: The new special bit to update.
        pos: The position to update the special bit at.
    """
    if not isinstance(carry_bits_dict, dict):
        raise TypeError("carry_bits_dict must be a dictionary.")
    if not isinstance(carry_bit_dict, dict):
        raise TypeError("carry_bit_dict must be a dictionary.")
    if not isinstance(pos, int):
        raise TypeError("pos must be an integer.")

    for key in carry_bit_dict.keys():
        if key == 'bit_len_used':
            continue
        assert key in carry_bits_dict, f"Key {key} not found in carry_bits_dict."
        debug_print(f"Updating carry_bits_dict[{key}] at position {pos} with value {carry_bit_dict[key]}")
        if pos >= len(carry_bits_dict[key]):
            while len(carry_bits_dict[key]) <= pos:
                carry_bits_dict[key].append(None)

        carry_bits_dict[key][pos] = carry_bit_dict[key]
        debug_print(f"Updated carry_bits_dict[{key}][{pos}] to {carry_bits_dict[key][pos]}")

def kleene_star_compacted_set_bitmaps(sp_marker: CompactBitSet, sp_indexes: list,
                                      matcher: Callable[[CompactBitSet, list[CompactBitSet], list[int], int, dict], Tuple[CompactBitSet, dict]],
                                      length: int = 1,
                                      carry_bits_dict: dict = None
                                      ) -> Tuple[CompactBitSet, dict]:
    """
    Perform a Kleene Star operation on a compacted bit stream.
    Args:
        sp_marker: The compacted bit stream to apply the Kleene Star operation on.
        sp_indexes: The index list of the compacted bit stream.
        matcher: A callable that performs the matching operation.
        length: The length of the bit streams.
        carry_bits: A dictionary containing carry bits.
    Returns:
        A CompactBitSet representing the result of the Kleene Star operation.
    """
    if not isinstance(sp_marker, CompactBitSet):
        raise TypeError("sp_marker must be a CompactBitSet instance.")
    if not isinstance(sp_indexes, list):
        raise TypeError("sp_indexes must be a list.")

    marker = CompactBitSet(bytearray(), [])
    marker.copy(sp_marker)
    marker_tmp = CompactBitSet(bytearray(), [])
    marker_tmp.copy(sp_marker)
    marker_set_tmp = CompactBitSet(bytearray(), [])

    carry_bits_dict_out = {k: type(v)() for k, v in carry_bits_dict.items()}
    debug_print(f"carry_bits_dict_out initialized: {carry_bits_dict_out}")
    iteration = 0
    while any(marker_tmp.marker) or iteration < carry_bits_dict['bit_len_used']:
        debug_print(f"Kleene Star iteration {iteration}: marker_tmp={marker_tmp.marker}, indexes={marker_tmp.indexes}")
        carry_bit_dict = get_dict_spc_pos(carry_bits_dict, iteration)
        marker_set_tmp, marker_tmp, carry_bit_dict = matcher(marker_tmp, [], sp_indexes, length, carry_bit_dict)
        debug_print(f"Marker after matcher: {marker_tmp.marker, marker_tmp.indexes}, carry_bit_dict={carry_bit_dict}")
        update_dict_spc_pos(carry_bits_dict_out, carry_bit_dict, iteration)

        if any(marker_set_tmp.marker):
            debug_print(f'new marker: {marker_set_tmp.marker} index: {marker_set_tmp.indexes}')
            marker.update(marker_set_tmp)
            debug_print(f'marker: {marker.marker} index: {marker.indexes}')
        iteration += 1

    carry_bits_dict_out['bit_len_used'] = iteration

    return CompactBitSet(marker.marker, marker.indexes), carry_bits_dict_out


def test_concat_compacted_set_bitmaps():
    """
    Test the concatenation of compacted bit streams.
    """
    sp_marker1 = CompactBitSet(bytearray([0b00000000, 0b11001100, 0b10101010, 0b10000000]), [0, 1, 2, 3])
    sp_marker2 = CompactBitSet(bytearray([0b00001111, 0b11110000]), [2, 3])
    sp_indexes = [0, 1, 2, 3]
    sp_carry_bit = CarryBit(index=-1, value=0)
    expect_result = CompactBitSet(bytearray([0b00001111, 0b10000000]), [2, 3])
    expect_carry_bit = CarryBit(index=3, value=1)
    expect_forward = CompactBitSet(bytearray([0b00000101]), [2])

    result, forward, carry_bit = concat_compacted_set_bitmaps(sp_marker1, sp_marker2, sp_indexes, 4, sp_carry_bit)

    assert result.marker == expect_result.marker, f"Expected marker: {expect_result.marker}, got: {result.marker}"
    assert result.indexes == expect_result.indexes, f"Expected indexes: {expect_result.marker}, got: {result.indexes}"
    assert carry_bit.index == expect_carry_bit.index, f"Expected carry index: {expect_carry_bit.index}, got: {carry_bit.index}"
    assert carry_bit.value == expect_carry_bit.value, f"Expected carry value: {expect_carry_bit.value}, got: {carry_bit.value}"
    assert forward.marker == expect_forward.marker, f"Expected forward marker: {expect_forward.marker}, got: {forward.marker}"
    assert forward.indexes == expect_forward.indexes, f"Expected forward indexes: {expect_forward.indexes}, got: {forward.indexes}"
    print("Test passed!")

def test_alt_compacted_set_bitmaps():
    """
    Test the alternation of compacted bit streams.
    """
    sp_marker1 = CompactBitSet(bytearray([0b00000000, 0b11001100, 0b10101010, 0b10000000]), [0, 1, 2, 3])
    sp_marker2 = CompactBitSet(bytearray([0b00001111, 0b11110000]), [2, 3])
    sp_indexes = [0, 1, 2, 3]

    result = alt_compacted_set_bitmaps(sp_marker1, sp_marker2, sp_indexes, 4)

    assert result.marker == bytearray([0b11001100, 0b10101111, 0b11110000]), f"Expected marker: {bytearray([0b10101111, 0b11110000])}, got: {result.marker}"
    assert result.indexes == [1, 2, 3], f"Expected indexes: [2, 3], got: {result.indexes}"
    print("Test passed!")

def test_kleene_star_compacted_set_bitmaps():
    """
    Test the Kleene Star operation on compacted bit streams.
    """
    sp_marker = CompactBitSet(bytearray([0b00000001, 0b10000000, 0b11001100, 0b10101010]), [0, 1, 2, 3])
    sp_indexes = [0, 1, 2, 3]

    def matcher(marker_tmp: CompactBitSet, _, indexes: list, length: int, carry_bit_dict: dict) -> Tuple[CompactBitSet, dict]:
        # Simple matcher that just returns the input marker and carry bits
        marker2 = CompactBitSet(bytearray([0b11110000, 0b00000000]), [2, 3])
        carry_bit_tmp = carry_bit_dict['test']
        concat_marker, concat_marker_forward, carry_bit_tmp = concat_compacted_set_bitmaps(marker_tmp, marker2, indexes, length, carry_bit_tmp)
        carry_bit_dict = {'bit_len_used': 1, 'test': carry_bit_tmp}
        return concat_marker, concat_marker_forward, carry_bit_dict

    result, carry_bits_out = kleene_star_compacted_set_bitmaps(sp_marker, sp_indexes, matcher, length=4, carry_bits_dict={'bit_len_used': 1, 'test': [CarryBit(-1, 0)]})
    expected_marker = CompactBitSet(bytearray([0b00000001, 0b10000000, 0b11111100, 0b10101010]), [0, 1, 2, 3])
    expceted_carry_bits = {'bit_len_used': 5, 'test': [CarryBit(3, 1)] + [CarryBit(3, 0) for _ in range(4)]}

    assert result.marker == expected_marker.marker, f"Expected marker: {expected_marker.marker}, got: {result.marker}"
    assert result.indexes == expected_marker.indexes, f"Expected indexes: {expected_marker.indexes}, got: {result.indexes}"
    assert carry_bits_out['bit_len_used'] == expceted_carry_bits['bit_len_used'], f"Expected bit_len_used: {expceted_carry_bits['bit_len_used']}, got: {carry_bits_out['bit_len_used']}"
    assert carry_bits_out['test'] == expceted_carry_bits['test'], f"Expected carry bits: {expceted_carry_bits['test']}, got: {carry_bits_out['test']}"
    print("Test passed!")

if __name__ == "__main__":
    test_concat_compacted_set_bitmaps()
    test_alt_compacted_set_bitmaps()
    print("\nTesting Kleene Star operation on compacted bit streams...")
    test_kleene_star_compacted_set_bitmaps()
    print("All tests passed!")