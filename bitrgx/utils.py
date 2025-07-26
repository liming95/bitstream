#1. character -> bit stream
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
def char_to_bit_stream(data: bytearray):
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
    return char_to_bit_stream(bytearray(data))
#2. bit stream -> bit stream for special characters (markers, etc.)
#3. compact sparse markers to index list and bit streams
#4. merge and sort index lists

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python utils.py <bytearray>")
        sys.exit(1)
    input_data = bytearray(sys.argv[1], encoding='ascii')
    if len(input_data) != 8:
        print("Error: Input must be exactly 8 bytes long.")
        sys.exit(1)
    try:
        result = char_to_bit_stream(input_data)
        print("Bit stream:", result)
    except (TypeError, ValueError) as e:
        print("Error:", e)