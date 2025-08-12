import itertools

# This code is a simplified example of a backward pass for a set-based operation.
_inst_id_gen = itertools.count(1)

def shift_right(s):
    return {x - 1 for x in s if x > 0}

def set_and_forward(s1, s2):
    R1 = s1 & s2
    R2 = {x + 1 for x in s1}
    R3 = R2 & s2
    return R1 | R3, R3

def set_and_backward(entry, needed_out):
    s1_val, s2_val = entry['in_vals']
    out_all_val, out_R3_val = entry['out_vals']
    R1 = s1_val & s2_val
    R3 = out_R3_val
    R1_dep = needed_out & R1
    R3_dep = needed_out & R3
    s1_dep = set(R1_dep) | shift_right(R3_dep)
    s2_dep = set(R1_dep) | set(R3_dep)
    return {'s1': s1_dep, 's2': s2_dep}

def set_or_forward(s1, s2):
    return s1 | s2

def set_or_backward(entry, needed_out):
    s1_val, s2_val = entry['in_vals']
    s1_dep = needed_out & s1_val
    s2_dep = needed_out & s2_val
    return {'s1': s1_dep, 's2': s2_dep}

def set_assign(s_in):
    return set(s_in)

def set_assign_backward(entry, needed_out):
    # assign: R = tmp_set  的逆向：需要哪些 tmp_set 的元素
    # entry['in_vals'] 单输入
    s_val = entry['in_vals'][0]
    s_dep = needed_out & s_val
    return {'s1': s_dep}

def forward_and_trace(b_set, a_set, n_set):
    trace = []
    last_assign = {}

    def add_inst(op, outputs, inputs, in_vals, out_vals):
        inst_id = next(_inst_id_gen)
        ins_with_id = [(name, last_assign.get(name, 0)) for name in inputs]
        outs_with_id = [(name, inst_id) for name in outputs]
        trace.append({
            'id': inst_id,
            'op': op,
            'outs': outs_with_id,
            'out_vals': out_vals,
            'ins': ins_with_id,
            'in_vals': in_vals
        })
        for name in outputs:
            last_assign[name] = inst_id
        return inst_id

    # inst1: produce tmp_set
    tmp_set, tmp_R3 = set_and_forward(b_set, a_set)
    inst1_id = add_inst('set_and',
             outputs=['tmp_set', 'tmp_R3'],
             inputs=['b_set', 'a_set'],
             in_vals=[set(b_set), set(a_set)],
             out_vals=[set(tmp_set), set(tmp_R3)])
    # bind initial R to the same producing inst as tmp_set is NOT done by set_assign yet;
    # we'll explicitly create an assign inst so R has its own inst id (helps tracing)
    R = set_assign(tmp_set)
    inst_assign_id = add_inst('set_assign',
             outputs=['R'],
             inputs=['tmp_set'],
             in_vals=[set(tmp_set)],
             out_vals=[set(R)])

    # while loop
    while tmp_set:
        tmp1, tmp = set_and_forward(tmp_set, n_set)
        inst2_id = add_inst('set_and',
                 outputs=['tmp1', 'tmp_tmp1_r3'],
                 inputs=['tmp_set', 'n_set'],
                 in_vals=[set(tmp_set), set(n_set)],
                 out_vals=[set(tmp1), set(tmp)])

        tmp2, tmp = set_and_forward(tmp1, b_set)
        inst3_id = add_inst('set_and',
                 outputs=['tmp2', 'tmp_tmp2_r3'],
                 inputs=['tmp1', 'b_set'],
                 in_vals=[set(tmp1), set(b_set)],
                 out_vals=[set(tmp2), set(tmp)])

        R_new = set_or_forward(R, tmp2)
        inst4_id = add_inst('set_or',
                 outputs=['R'],
                 inputs=['R', 'tmp2'],
                 in_vals=[set(R), set(tmp2)],
                 out_vals=[set(R_new)])
        R = R_new
        tmp_set = tmp

    return R, trace, last_assign

def backward_pruned(trace, R_final, last_assign, input_names):
    # needed: key (var, inst_id) -> set of needed elements of that particular value
    needed = {}
    deps = {name: set() for name in input_names}

    # initialize needed with the final R
    r_inst = last_assign.get('R', None)
    if r_inst is None:
        raise RuntimeError("No R in last_assign; cannot start backward")
    needed[('R', r_inst)] = set(R_final)

    # backward trace
    for entry in reversed(trace):
        # check if this entry is relevant and calculate needed outputs
        out_needed = set()
        for (name, oid), val in zip(entry['outs'], entry['out_vals']):
            needed_val = needed.get((name, oid))
            if needed_val:
                intersect = needed_val & val
                if intersect:
                    out_needed |= intersect

        if not out_needed:
            continue

        if entry['op'] == 'set_and':
            in_deps = set_and_backward(entry, out_needed)  # expects {'s1','s2'}
            # entry['ins'] has two inputs
            (in1_name, in1_id), (in2_name, in2_id) = entry['ins']
            # if there are multi backward path, update needed for both inputs
            needed[(in1_name, in1_id)] = needed.get((in1_name, in1_id), set()) | in_deps['s1']
            needed[(in2_name, in2_id)] = needed.get((in2_name, in2_id), set()) | in_deps['s2']
            # If the input names are in deps, update their dependencies
            if in1_name in deps:
                deps[in1_name] |= in_deps['s1']
            if in2_name in deps:
                deps[in2_name] |= in_deps['s2']

        elif entry['op'] == 'set_or':
            in_deps = set_or_backward(entry, out_needed)
            (in1_name, in1_id), (in2_name, in2_id) = entry['ins']
            needed[(in1_name, in1_id)] = needed.get((in1_name, in1_id), set()) | in_deps['s1']
            needed[(in2_name, in2_id)] = needed.get((in2_name, in2_id), set()) | in_deps['s2']
            if in1_name in deps:
                deps[in1_name] |= in_deps['s1']
            if in2_name in deps:
                deps[in2_name] |= in_deps['s2']

        elif entry['op'] == 'set_assign':
            # sigle assign: outputs=['R'], inputs=['tmp_set']
            in_deps = set_assign_backward(entry, out_needed)  # returns {'s1':...}
            (in1_name, in1_id) = entry['ins'][0]
            needed[(in1_name, in1_id)] = needed.get((in1_name, in1_id), set()) | in_deps['s1']
            if in1_name in deps:
                deps[in1_name] |= in_deps['s1']

        else:
            # Todo: backward handler for unsupported operations
            print(f"Warning: Unsupported operation {entry['op']} in backward pass; skipping")

    return deps

# TEST
b_set = {0, 1, 4, 5, 6, 7}
a_set = {0, 3, 4, 7}
n_set = {3, 4, 7}
R, trace, last_assign = forward_and_trace(b_set, a_set, n_set)
deps = backward_pruned(trace, R, last_assign, ['b_set', 'a_set', 'n_set'])

print("R =", R)
print("deps =", deps)
print("\nTrace:")
for t in trace:
    print(t)
