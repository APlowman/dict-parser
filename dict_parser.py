import numpy as np
import copy
import formatting


def get_from_dict(d, address, is_idx, default=None):
    """
    Retrive an object from a dict given an address.

    Parameters
    ----------
    d : dict
        Dict containing other nested dicts, lists and Numpy arrays.
    address : list of length n
        List representing the address of the object to retrieve in the dict.
    is_idx : list of bool of length n
        List denoting whether the corresponding address element is a (list or
        array) index (True) or a dict key (False).
    default : 
        Object to return if address does not resovle.

    Returns
    -------
    object

    """

    if len(address) != len(is_idx):
        raise ValueError(
            'Length of `address` ({}) and length of `is_idx` ({}) '
            'do not match.'.format(len(address), len(is_idx)))

    if not np.all([isinstance(i, bool) for i in is_idx]):
        raise ValueError('`is_idx` must be a list of bools.')

    if not np.all(
        [isinstance(address[i], int)
         for i in [idx for idx, i in enumerate(is_idx) if i == True]]):
        raise ValueError('Elements in `address` which correspond to `True` in '
                         '`is_idx` should be integers.')

    subobj = d
    for (k_idx, k), idx in zip(enumerate(address), is_idx):

        if not idx:
            try:
                subobj = subobj.get(k)

            except KeyError:
                subobj = default
                break

            except AttributeError:

                path = '->'.join(['"{}"'.format(i) for i in address[:k_idx]])
                raise AttributeError(
                    'Address {} does not resolve to a dict, but `is_idx[{}]` '
                    'is False.'.format(path, k_idx))

        else:
            try:
                subobj = subobj[k]

            except IndexError:
                subobj = default
                break

    return subobj


def set_in_list(a, idx, val):
    """
        Set an element to value `val` in a nested list `a`. The position
        of the element to set is given by the list `idx`.

        Example:

        `a` = [[1, 2], [3, [4, 5], 6], [7]]
        `idx` = [1,1,0]
        `val` = 9

        ==> `a` = [[1, 2], [3, [9, 5], 6], [7]]

    """

    for i in range(len(idx) - 1):

        a = a[idx[i]]

    a[idx[-1]] = val


def get_consec_arr_row_idx(shape):
    """
        Given Numpy array shape given by the tuple `shape`,
        return an array of row vectors, each of which represent the 
        indices along each dimension of consecutive rows of the array
        (i.e. the number of columns in return array is len(shape) - 1).

        Example:

            `shape` = (2,3,2)

        returns:

            [[0 0]
             [0 1]
             [0 2]
             [1 0]
             [1 1]
             [1 2]]        

    """

    idx_col = []
    cum_prod = [np.prod(shape[len(shape) - i - 1:-1])
                for i in range(len(shape))]

    for i_idx, i in enumerate(range(len(shape) - 2, -1, -1)):

        num_tiles = cum_prod[-1] / cum_prod[i_idx + 1]
        idx_col.append(np.tile(
            np.repeat(
                np.arange(shape[i]),
                cum_prod[i_idx]),
            int(num_tiles))[:, np.newaxis])

    idx_all = np.hstack(idx_col)[:, ::-1]

    return idx_all


def set_in_dict(obj, address, is_idx, val):
    """

    Can append list elements with setting `address` element to be `None`
    Do not concatenate array elements.

    """

    orig_obj = copy.deepcopy(obj)
    obj = val
    idx = len(address) - 1

    while idx >= 0:

        # idx decreases during while loop

        nested_idx_num = 0
        is_cur_idx = is_idx[idx]

        while is_cur_idx:

            # Keep moving backwards along is_idx until we get to a dictionary

            nested_idx_num += 1
            idx -= 1
            is_cur_idx = is_idx[idx]

        if nested_idx_num > 0:

            # Retrieve subobj:
            subobj = get_from_dict(
                orig_obj, address[:idx + 1], is_idx[:idx + 1])

            subsubobj = None

            if address[idx + nested_idx_num] == None:

                # Last element in address is None => need to append (list)

                subsubobj = subobj

                for subidx in range(idx + 1, idx + nested_idx_num):

                    subsubobj = subsubobj[address[subidx]]

                if isinstance(subsubobj, list):
                    subsubobj.append(None)
                    new_idx = len(subsubobj) - 1

                address[idx + nested_idx_num] = new_idx

            set_in_list(
                subobj, address[idx + 1: idx + 1 + nested_idx_num], obj)

            obj = subobj

        subobj = {address[idx]: obj}

        obj = copy.deepcopy(get_from_dict(
            orig_obj, address[:idx], is_idx[:idx]))

        obj = {k: v for k, v in obj.items() if k != address[idx]}
        obj.update(subobj)

        idx -= 1

    return obj


def parse_string_as(val, data_type):
    """ 
        Parse a string as int, float, bool or string. Floats may be expressed 
        as a string like "34/2.3"

    """
    if isinstance(val, str):

        try:

            parsed_val = False

            if data_type is object or data_type is str:
                parsed_val = val

            elif data_type == int:
                parsed_val = int(val)

            elif data_type is float:

                if "/" in val:
                    num, den = val.split("/")
                    parsed_val = float(num) / float(den)

                else:
                    parsed_val = float(val)

            elif data_type is bool:

                if val.upper() in ['TRUE', 'YES', '1', '1.0']:
                    parsed_val = True

                elif val.upper() in ['FALSE', 'NO', '0', '0.0']:
                    parsed_val = False

                else:
                    raise ValueError(
                        'Cannot parse string {} as type bool'.format(val))

            else:
                raise ValueError(
                    'Cannot parse string {} as type {}'.format(val, data_type))

        except ValueError:
            raise

        return parsed_val

    else:
        raise ValueError('Value passed to parse_string_as ({}) is not a'
                         'string.'.format(val))


def parse_dict_str(s):
    """
    Parse a string as a dict containing nested dicts, lists and Numpy arrays.

    """

    ASSIGN_SYM = '='
    LIST_OPEN = '['
    LIST_CLOSE = ']'
    DICT_OPEN = '{'
    DICT_CLOSE = '}'
    ARR_OPEN = '*['
    ARR_CLOSE = ']'
    DTYPE_OPEN = '('
    DTYPE_CLOSE = ')'

    ALL_OPEN = [LIST_OPEN, DICT_OPEN, ARR_OPEN]
    ALL_CLOSE = [LIST_CLOSE, DICT_CLOSE, ARR_CLOSE]

    ln_s = s.split('\n')

    out = {}
    address = []
    is_idx = []

    OPEN_OBJ_DEF = {
        '[': [],
        '{': {}
    }

    ARR_DTYPES_DEF = {
        'int': int,
        'float': float,
        'bool': bool
    }

    # Loop through lines twice, first pass to get shape of any arrays,
    # second to parse data.

    arr_shapes = []
    arr_dtypes = []
    cur_arr_consec_blocks = []
    cur_arr_consec_blanks = []
    parse_consec_blanks = False
    parse_consec_blocks = False
    arr_start = False
    parse_arr_shape = False

    # First loop, to get array shapes
    for ln_idx, ln in enumerate(ln_s):

        ln = ln.strip()

        if ASSIGN_SYM in ln:
            k, v = [i.strip() for i in ln.split(ASSIGN_SYM)]
            dtype_str = None

            if DTYPE_OPEN in v and DTYPE_CLOSE in v:
                dtype_str, v = v.split(DTYPE_OPEN)[1].split(DTYPE_CLOSE)

            if v == ARR_OPEN:
                arr_start = True
                arr_dtypes.append(ARR_DTYPES_DEF.get(dtype_str, None))

        elif ln == ARR_OPEN:
            arr_start = True

        elif ln == ARR_CLOSE and parse_arr_shape:
            parse_arr_shape = False

            # Check all inner array blocks have the same dimensions:
            all_inner_agree = all(
                [cur_arr_consec_blocks[0] == cur_arr_consec_blocks[i]
                 for i in range(1, len(cur_arr_consec_blocks))])

            if not all_inner_agree:
                raise ValueError(
                    'Malfored array on line number {}: not all array blocks '
                    'have the same dimensions.'.format(ln_idx))

            # Based on the list of the number of consecutive blank lines
            # between array blocks, work out the outer dimensions of the array.
            # The inner two dimensions are already computed.

            cur_arr_shape = []

            if len(cur_arr_consec_blanks) > 0:

                # Get max number of consecutive blank lines:
                max_blanks = max(cur_arr_consec_blanks)
                num_i = 1

                for i in range(max_blanks, 0, -1):

                    if i not in cur_arr_consec_blanks:

                        raise ValueError(
                            'Malformed array on line number {}: '
                            'Some blocks are separated by {} blank lines,'
                            'but there are no blocks separated by {} blank '
                            'lines.'.format(ln_idx, max_blanks, i))

                    num_i = 1 + (cur_arr_consec_blanks.count(i) / num_i)
                    cur_arr_shape.append(int(num_i))

            cur_arr_shape.extend(cur_arr_consec_blocks[0])
            arr_shapes.append(tuple(cur_arr_shape))

            cur_arr_consec_blocks = []
            cur_arr_consec_blanks = []
            parse_consec_blocks = False

        if arr_start:
            arr_start = False
            parse_arr_shape = True
            continue

        if parse_arr_shape:

            if ln == '':
                if not parse_consec_blanks:
                    parse_consec_blanks = True
                    cur_arr_consec_blanks.append(1)

                else:
                    cur_arr_consec_blanks[-1] += 1

                if parse_consec_blocks:
                    parse_consec_blocks = False

            else:
                width = len(ln.split())

                if parse_consec_blanks:
                    parse_consec_blanks = False

                if not parse_consec_blocks:
                    parse_consec_blocks = True
                    cur_arr_consec_blocks.append([1, width])

                else:
                    cur_arr_consec_blocks[-1][0] += 1
                    prev_width = cur_arr_consec_blocks[-1][1]

                    if width != prev_width:
                        raise ValueError(
                            'Malformed array on line number {}. Expected {} '
                            'columns, but found {}.'.format(
                                ln_idx, prev_width, width))

    arr_row_idx = [get_consec_arr_row_idx(i) for i in arr_shapes]

    depth_change = 1
    new_is_idx_elm = False
    arr_ln_idx = -1
    is_arr = False
    arr_idx = -1
    dtype_str = None

    # Second loop, to parse data:
    for ln in ln_s:

        ln = ln.strip()

        if ln == '':
            continue

        k, v, idx = None, None, None
        open_obj, open_str = None, None

        # Set the line type:
        if ASSIGN_SYM in ln:

            dtype_str = None

            k, v = [i.strip() for i in ln.split(ASSIGN_SYM)]

            if DTYPE_OPEN in v and DTYPE_CLOSE in v:
                dtype_str, v = v.split(DTYPE_OPEN)[1].split(DTYPE_CLOSE)

            if any([v == i for i in ALL_OPEN]):
                ln_type = 'ASSIGN_OPEN'
                open_str = v
                open_obj = OPEN_OBJ_DEF.get(v, None)

            else:
                ln_type = 'ASSIGN'

        elif any([ln == i for i in ALL_OPEN]):
            ln_type = 'OPEN'
            open_str = ln
            open_obj = OPEN_OBJ_DEF.get(ln, None)

        elif any([ln == i for i in ALL_CLOSE]):
            ln_type = 'CLOSE'

        else:
            ln_type = 'VALUE'
            v = ln.strip()

        # Parse data on line:
        if ln_type == 'CLOSE':

            address = address[:-arr_row_idx[arr_idx].shape[1]]
            is_idx = is_idx[:-arr_row_idx[arr_idx].shape[1]]
            is_arr = False
            arr_ln_idx = -1

        else:

            # Find new address element:
            if ln_type in ['ASSIGN_OPEN', 'ASSIGN']:
                new_address_elem = k

            elif ln_type in ['OPEN', 'VALUE']:
                new_address_elem = idx

            # Set new address element:
            if depth_change == 0:
                address[-1] = new_address_elem

            elif depth_change == 1:
                address.append(new_address_elem)
                is_idx.append(new_is_idx_elm)

            # Find depth change of the next line relative to current line:
            if ln_type in ['ASSIGN_OPEN', 'OPEN']:
                depth_change = 1

            elif ln_type in ['ASSIGN', 'VALUE']:
                depth_change = 0

            # Set a new value:
            if ln_type == 'ASSIGN':

                if dtype_str is None:
                    set_val = v

                else:
                    set_val = parse_string_as(v, ARR_DTYPES_DEF[dtype_str])

            elif ln_type == 'VALUE':
                set_val = v

            elif ln_type in ['ASSIGN_OPEN', 'OPEN']:
                set_val = open_obj

            if open_str == ARR_OPEN:

                arr_idx += 1
                set_val = np.zeros(
                    arr_shapes[arr_idx], dtype=arr_dtypes[arr_idx])
                is_arr = True

            if is_arr:

                if arr_ln_idx >= 0:

                    # Parse an array row
                    set_val = [i for i in set_val.split()]

                    if arr_ln_idx == 0:
                        address = address[:-1] + \
                            list(arr_row_idx[arr_idx][arr_ln_idx])
                        is_idx = is_idx[:-1] + [True, ] * \
                            arr_row_idx[arr_idx].shape[1]

                    else:
                        address[-arr_row_idx[arr_idx].shape[1]:] = list(
                            arr_row_idx[arr_idx][arr_ln_idx])

                        is_idx[-arr_row_idx[arr_idx].shape[1]:] = [
                            True, ] * arr_row_idx[arr_idx].shape[1]

                arr_ln_idx += 1

            out = set_in_dict(out, address, is_idx, set_val)

            # For the next line:
            new_is_idx_elm = True if open_str == LIST_OPEN else False

    return out
