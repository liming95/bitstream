#!/usr/bin/python

print ("Hello, World!") 
MASK = (1 << 32) -1 
def logical_right_shift(value, shift, bits=32):
    mask = (1 << bits) - 1
    return (value & mask) >> shift

def remove_rightmost_1(value):
    return value & (value - 1)

def extract_rightmost_1(value):
    return value & ((~value) + 1)

def extract_and_smear_rightmost_1(value):
    return value ^ (value - 1)

def extract_leftmost_1block(value):
    backslash_block = value
    backslash_final = 0
    block_num = 0
    while backslash_block != 0:
        backslash_final = backslash_block
        block_num = block_num + 1
        backslash_block_first_start = extract_and_smear_rightmost_1(backslash_block)
        backslash_block_first = backslash_block & backlash_block_first_start
        backslash_block_first_end = extract_and_smear_rightmost_1(~backslash_block_first) 
        backslash_block = backslash_block_first & (~backslash_block_first_end)
    return (backslash_final, block_num)

class PatternForest:
    # It is possible that pattern include nested fields
    def __init__(self, pattern):
        # 1. covert pattern to pattern tree
        self.tree_num = len(pattern)
        self.forest = {}
        forest = pattern.iterms()
        for i in range(self.tree_num):
            (name, node_list) = forest[i]
            pattern_tree = PatternTree(node_list)
            self.forest[name] = [pattern_tree]
    
    def insert_pattern(self, pattern):
        assert self.tree_num == len(pattern)
        forest = pattern.iterms()
        for i in range(self.tree_num):
            (name, node_list) = forest[i]
            assert sel.forest.has_key(name)
            pattern_tree = self.forest[name]
            pattern_tree.insert_pattern(node_list)

    class PatternTree:
        def __init__(self, pattern):
            self.level_num = len(pattern)
            self.node_in_level = []
            self.root = PatternNode()
            for i in range(self.level_num):
                node = PatternNode(pattern[i])
                self.node_in_level.append([node])
                node.children.append(0)
            del self.node_in_level[self.level_num-1][0].children[0]
            self.root.children.append(0)
        
        def insert_pattern(self, pattern):
            assert self.level_num == len(pattern)
            parent = self.root
            for i in range(self.level_num):
                (name, index) = pattern[i]
                insert_flag = False
                # merge new node to original node in the tree
                for child in parent.children:
                    node_in_tree = self.node_in_level[child]
                    if node_in_tree.name == name and node_in_tree.index == index:
                        node_in_tree.weight = node_in_tree.weight + 1
                        children = node_in_tree
                        insert_flag = True
                        break
                # insert new node to the tree
                if insert_flag == False:
                    node = PatternNode(pattern[i])
                    self.node_in_level[i].append(node)
                    parent.children.append(len(self.node_in_level[i])-1)
                    parent = node 

    class PatternNode:

        def __init__(self, field = None):
            if field is not None:
                self.name = field[0]
                self.index = field[1]
                self.weight = 1
            self.children = []

class MisonParser:

    training_number = 5
    bitmap_word_size = 32
    def __init__(self, text, field_query):
        self.text = text
        self.field = field_query # hash_table
        self.value = []
        self.pattern_field = split_field(field)
        self.pattern_tree = None

    def split_field(field_list):
        pattern_field = []
        level = 0
        for field_name in self.field:
            nested_level = len(split(field_name, '.'))
            level = level > nested_level \
                ? level : nested_level
        
        for i in range(level):
            pattern_field.append({})
        
        for field in field_list:
            field_layer = split(field, '.')
            field_name = "root"
            for i in range(len(field_layer)):
                field_name = field_name + '.' + field_layer[i]
                if i == len(field_layer) - 1:
                    pattern_field[i][field_name] = True
                else:
                    pattern_field[i][field_name] = False
    
    def display_fields(self):
        # 1.training_phase
        # 2.travel all the reminding json record
        # 3.return the result of query in the text
        training_phase(training_number)
        for i in range(training_number, len(self.text)):
            parsing_phase(self.text[i])
        return self.value

    def training_phase(self, num):
        # num: the number of records used to build the pattern tree
        # 1. build colon index bitmap for each training json record
        # 2. parse queried fields according to the colon bitmap
        # 3. build pattern tree (fields, logic positions)
        for i in range(training_number):
            record = self.text[i]
            bitmap = get_bitmap_colon(record)
            # pattern = {"root":[(field, positions),...], "root.nested_field":[(field, position),...],...}
            pattern = parser_training(record, bitmap) 
            collect_pattern(pattern)
    
    def get_bitmap_colon(self, record_json):
        # return structural colon bitmap
        # 1.travel json record to build bitmaps for "\", """, ":", "{", "}"
        
        num_ints = (len(record_json) + bitmap_word_size-1) / bitmap_word_size
        backslash_bitmap = [0] * num_ints
        quote_bitmap = [0] * num_ints
        string_bitmap = [0] * num_ints
        colon_bitmap = [0] * num_ints
        left_brace = [0] * num_ints
        right_brace = [0] * num_ints
        

        for i, char in enumerate(record_json):
            int_index = i / bitmap_word_size
            bit_position = 1 << (i % bitmap_word_size)

            if char == "\\":
                backslash_bitmap[int_index] |= bit_position
            if char == "\"":
                quote_bitmap[int_index] |= bit_position
            if char == ":":
                colon_bitmap[int_index] |= bit_position
            if char == "{":
                left_brace_bitmap[int_index] |= bit_position
            if char == "}":
                right_brace_bitmap[int_index] |= bit_position
        
        # 2.build structural """ bitmap
        for i in range(num_ints):
            if i != num_ints - 1:
                pro_escaped_char_bitmap = \
                    backslash_bitmap[i] & logical_right_shift(quote_bitmap[i], 1) \
                    &  (quote_bitmap[i+1] << (bitmap_word_size-1)) & MASK
            else:
                pro_escaped_char_bitmap = \
                    backslask_bitmap[i] & logical_right_shift(quote_bitmap[i], 1)
            
            escaped_char_bitmap = pro_escaped_char_bitmap
            while pro_escaped_char_bitmap != 0:
                first_escaped_char = extract_and_smear_rightmost_1(pro_escaped_char_bitmap)
                backslash_block = backslash_bitmap[i] & first_escaped_char
                # method1:using for to travel the consecutive backslash characters is less 32 times
                # method2:reverse the backslash and get the position of first 0, and then get the consecutive characters,and then use popcnt
                # method3:
                (backslash_final, block_num) = extract_leftmost_1block(backslash_block)
                backslash_num = backslash_final.bit_count()
                backslash_index = i
                #backslash is containing in the last item of backslash bitmap
                while block_num == 1 and ((backslash_final & 0x1) == 0x1):
                    assert backslash_index != 0
                    if (backslash_bitmap[backslash_index-1] & (1 << (bitmap_word_size-1))) != 0:
                        (backslash_final, block_num) = extract_leftmost_1block(backslash_bitmap[backslash_index-1])
                        backslash_num = backslash_num + backslash_final.bit_count()
                    else:
                        break
                if backslash_num % 2  == 0:
                    escaped_char_bitmap = escaped_char_bitmap ^ extract_rightmost_1(pro_escaped_char_bitmap)
                pro_escaped_char_bitmap = remove_rightmost_1(pro_escaped_char_bitmap)
            quote_bitmap[i] = quote_bitmap[i] ^ (escaped_char_bitmap << 1)
            if i != num_ints -1:
                quote_bitmap[i+1] = quote_bitmap[i+1] ^ logical_right_shift(quote_bitmap[i+1])
         
        # 3.build string bitmap
        quote_num = 0
        for i in range(num_ints):
            quote_bit = quote_bitmap[i]
            string_bit = 0
            while quote_bit != 0:
                first_quote_char = extract_and_smear_rightmost_1(quote_bit)
                string_bit = string_bit ^ first_quote_char
                quote_bit = remove_rightmost_1(quote_bit)
                quote_num = quote_num + 1
            
            if quote_num % 2 == 1:
                string_bitmap[i] = ~string_bit
            else:
                string_bitmap[i] = string_bit
        
    
        # 4.build structural colon bitmap
        for i in range(num_ints):
            colon_bitmap[i] = colon_bitmap[i] ^ (colon_bitmap[i] & string_bitmap[i])
            right_brace_bitmap[i] = right_brace_bitmap[i] ^ (colon_bitmap[i] & string_bitmap[i])
            left_brace_bitmap[i] = left_brace_bitmap[i] ^ (left_brace_bitmap[i] & string_bitmap[i])
        
        # 5.build leveled colon bitmap
        colon_level = 0
        for field_name in self.field:
            nested_level = len(split(field_name, '.'))
            colon_level = colon_level > nested_level \
                ? colon_level : nested_level
        
        colon_level_bitmap = []
        for i in range(colon_level):
            colon_level_bitmap.append(colon_bitmap[:])
        
        stack = []
        for i in range(num_ints):
            left_brace_bit = left_brace_bitmap[i]
            right_brace_bit = right_brace_bitmap[i]
            
            while True:
                first_right_brace_char = extract_rightmost_1(right_brace_bit)
                first_left_brace_char = extract_rightmost_1(left_brace_bit)
                while (first_left_brace_char != 0) and ((first_right_brace_char == 0) \
                    or (first_left_brace_char < first_right_brace_char)):
                    stack.append((i, first_left_brace_char))
                    left_brace_bit = remove_rightmost_1(left_brace_bit)
                    first_left_brace_char = extract_rightmost_1(left_brace_bit)

                if first_right_brace_char != 0:
                    (left_brace_word_index, left_brace_index) = stack.pop()
                    stack_depth = len(stack)
                    if stack_depth > 0 and stack_depth < colon_level:
                        if i == left_brace_word_index:
                            colon_level_bitmap[stack_depth-1][i] = \
                                colon_level_bitmap[stack_depth-1][i] & \
                                (~(first_right_brace_char - left_brace_index))
                        else:
                            colon_level_bitmap[stack_depth-1][left_brace_word_index] = \
                                colon_level_bitmap[stack_depth-1][left_brace_word_index] \
                                & (left_brace_index - 1)
                            colon_level_bitmap[stack_depth-1][i] = \
                                colon_level_bitmap[stack_depth-1][i] & (first_right_brace_char)
                            for j in range(left_brack_word_index+1, i):
                                colon_level_bitmap[stack_depth-1][j] = 0
                
                right_brace_bit = remove_rightmost_1(right_brace_bit)
                if (right_brace_bit == 0) and (left_brace_bit == 0):
                    break
        
        assert len(stack) == 0

        # That the level bitmap is not clear is also ok.
        for i in range(1, colon_level):
            for j in range(num_ints):
                colon_level_bitmap[colon_level-i][j] = \
                    colon_level_bitmap[colon_level-i-1][j] ^ \
                    colon_level_bitmap[colon_level-i][j]
        
        return colon_level_bitmap

    def parser_training(self, record_json, bitmap_level_colon):
        # 1. resolve the queried fields 
        # 2. return <fields, positions>
        pattern = parser_record(record_json, bitmap_level_colon)
        return pattern


    def collect_pattern(self, field_position_pairs):
        # 1. identify the type of field_position_pairs 
        # (missing, out_of_order, mixed)
        # 2. build nodes for the pairs
        # 3. insert this pattern into pattern tree
        if self.pattern_tree is None:
            self.pattern_tree = PatternTree(field_position_pairs)
        else:
            self.pattern_tree.insert_pattern(field_position_pairs)

    def parser_record(self, record_json, bitmap_level_colon, \
        start_index=0, end_index=len(record_json)-1, level=0, pre_field_name="root"):
        # resolve quried field from record_json
        # 1. calculate the physical index according to bitmap
        # 2. resolve the field name
        # 3. if field name is in the self.query, record the value and position
        # 4. else continue
        # store the result to the self.value
        assert level <= len(bitmap_level_colon)
        pattern = {pre_field_name:[]}
        logic_position = 0
        for i in range (start_index/bitmap_word_size, end_index/bitmap_word_size+1):
            colon_bit_word = bitmap_level_colon[level][i]
            if i == start_index/bitmap_word_size:
                bit_position = start_index % bitmap_word_size
                mask_clear = ~(extract_and_smear_rightmost_1(1 << bit_position))
                colon_bit_word = colon_bit_word & mask_clear
            elif i == end_index/bitmap_word_size:
                bit_position = end_index % bitmap_word_size
                mask_clear = extract_rightmost_1(1 << bit_position) - 1
                colon_bit_word = colon_bit_word & mask_clear 

            while colon_bit_word != 0:
                logic_position = logic_position + 1
                first_colon_bit_block = extract_and_smear_rightmost_1(colon_bit_word)
                colon_physical_index = bitmap_2_index(i, first_colon_bit_block)
                
                field_name = get_field_name(record_json, colon_physical_index-1)
                field_key = pre_field_name+'.'+field_name
                if field_key in self.pattern_field[level]:
                    pattern[pre_field_name].append((field_name, logic_position))
                    if self.pattern_field[level][field_key] == True:
                        field_value = get_field_value(record_json, colon_physical_index-1)
                        self.value.append(field_key, field_value) # the format of field_key is root.level1_field...
                    else:
                        # resolve the pattern of nested object
                        start_index_nested = colon_physical_index
                        end_index_nested = 0
                        remainder_colon_bitmap = remove_rightmost_1(colon_bit_word)
                        next_colon_word_index = i
                        while True:
                            # TO DO if condition
                            if remainder_colon_bitmap != 0:
                                first_colon_bit_block = extract_and_smear_rightmost_1(remainder_colon_bitmap)
                                end_index_nested = bitmap_2_index (next_colon_word_index, remainder_colon_bitmap)
                                break
                            elif next_colon_word_index < end_index/bitmap_word_size:
                                next_colon_word_index = next_colon_word_index + 1
                                remainder_colon_bitmap = bitmap_level_colon[level][next_colon_word_index]
                                if next_colon_word_index == end_index/bitmap_word_size:
                                    remainder_colon_bitmap = remainder_colon_bitmap & \
                                        ((1 << (end_index % bitmap_word_size)) - 1)
                                
                            elif next_colon_word_index == end_index/bitmap_word_size:
                                end_index_nested = end_index
                         
                        pattern_nested = parser_record(record_json,bitmap_level_colon,\
                            start_index_nested, end_index_nested, level+1, field_key)
                        pattern.update(pattern_nested)
                colon_bit_word = remove_rightmost_1(colon_bit_word)
        
        return pattern

    def bitmap_2_index(self, index_ints, bit_char, size=32):
        return index_ints * size + index_char.bit_count()
    
    def get_field_name(self, record_json, colon_physical_index):
        start_index = end_index = colon_physical_index - 1
        while (record_json[start_index] != ',') and (record_json[start_index] != '{'):
            assert satart_index >= 0
            start_index = start_index - 1
        
        field_name = record_json[start_index+1:end_index+1]
        return field_name
    
    def get_field_value(self, record_json, colon_physical_index):
        start_index = end_index = colon_physical_index + 1
        while (record_json[end_index] != ',') and (record_json[end_index] != '}'):
            assert end_index < len(record_json)
            end_index = end_index + 1
        
        field_value = record_json[start_index: end_index]
        return field_value

    def parsing_phase(self, record_json):
        # 1.get_bitmap_colon(record_json)
        # 2.speculate the field by combining pattern tree and colon bitmap
        # 3.if failed speculation, resolve fields by basic parser
        # 4.return fields

    def parser_speculation(self, record_json, bitmap_colon):
        # 1. covert logic position(pattern tree) to physical index(bitmap) for each field 
        # 2. check if the fields from the json text is matching the query field.
        # 3. if it is unequal, using the next pattern until all patterns used
        # 4. return the result of quired result and whether it is right

