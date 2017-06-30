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

    See also
    --------
    set_in_dict

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


def set_in_list(lst, idx, val):
    """
    Set an element in a nested list of any depth.

    Parameters
    ----------
    lst : list
        Nested list to be modified.
    idx : list of int
        Indices of the list, defining the position of the element to be
        modified. Each list element in `idx` correpsonds to an additional
        nesting depth.

    Returns
    -------
    None

    Examples
    --------
    >>> a = [[1, 2], [3, [4, 5], 6], [7], 8]
    >>> set_in_list(a, [1, 1, 0], -99)
    >>> a
    [[1, 2], [3, [-99, 5], 6], [7], 8]

    """

    for i in range(len(idx) - 1):
        lst = lst[idx[i]]

    lst[idx[-1]] = val


def get_consec_arr_row_idx(shape):
    """
    Get the indices for consecutive rows (innermost subarrays) of an array.

    Parameters
    ----------
    shape : tuple of length n
        Shape of the array for which indices should be returned.

    Returns
    -------
    ndarray
        Array of row vectors with n - 1 columns

    Examples
    --------
    >>> get_consec_arr_row_idx((2, 3, 2))
    [[0 0]
     [0 1]
     [0 2]
     [1 0]
     [1 1]
     [1 2]]

    """

    ndim = len(shape)
    idx_col = []
    cum_prod = [np.prod(shape[ndim - i - 1:-1]) for i in range(ndim)]

    for i_idx, i in enumerate(range(ndim - 2, -1, -1)):

        num_tiles = cum_prod[-1] / cum_prod[i_idx + 1]
        idx_col.append(np.tile(np.repeat(np.arange(shape[i]), cum_prod[i_idx]),
                               int(num_tiles))[:, np.newaxis])

    idx_all = np.hstack(idx_col)[:, ::-1]

    return idx_all


def set_in_dict(d, address, is_idx, val):
    """
    Set an object in a dict given an address.

    Parameters
    ----------
    d : dict
        Dict containing other nested dicts, lists and Numpy arrays.
    address : list of length n
        List representing the address of the object to set in the dict. If an
        element in this list is `None`, `val` will be appended to the list.
    is_idx : list of bool of length n
        List denoting whether the corresponding address element is a (list or
        array) index (True) or a dict key (False).
    val
        Object to set in dict `d` at address given by `address`.

    Returns
    -------
    dict
        A modified copy of the original dict.

    See also
    --------
    get_from_dict

    """

    orig_d = copy.deepcopy(d)
    d = val
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
            subobj = get_from_dict(orig_d, address[:idx + 1], is_idx[:idx + 1])

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
                subobj, address[idx + 1: idx + 1 + nested_idx_num], d)

            d = subobj

        subobj = {address[idx]: d}
        d = copy.deepcopy(get_from_dict(orig_d, address[:idx], is_idx[:idx]))
        d = {k: v for k, v in d.items() if k != address[idx]}
        d.update(subobj)
        idx -= 1

    return d


def parse_string_as(val, data_type):
    """
    Parse a string as an `int`, `float`, or `bool`.

    Parameters
    ----------
    val : str
        String to be parsed. If a value is to be parsed as a float, strings
        like '34/2.3' allowed.
    data_type : type
        One of `int`, `float`, `bool` or `str`.

    Returns
    -------
    parsed val

    Examples
    --------
    >>> parse_string_as('2.5/0.5', float)
    5.0

    """

    bool_strs = {
        True: ['TRUE', 'YES', '1', '1.0'],
        False: ['FALSE', 'NO', '0', '0.0']
    }

    if isinstance(val, str):

        try:
            parsed_val = False

            if data_type is object or data_type is str:
                parsed_val = val

            elif data_type is int:
                parsed_val = int(val)

            elif data_type is float:

                if "/" in val:
                    num, den = val.split("/")
                    parsed_val = float(num) / float(den)
                else:
                    parsed_val = float(val)

            elif data_type is bool:

                v_up = val.upper()
                if v_up in bool_strs[True]:
                    parsed_val = True
                elif v_up in bool_strs[False]:
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


def parse_dict_file(path):
    """
    Parse a text file as a dict containing nested dicts, lists and Numpy
    arrays.

    Parameters
    ----------
    path : str
        File path of a text file whose contents represents nested dicts, lists
        and Numpy arrays according to a particular syntax.

    Returns
    -------
    dict

    """

    with open(path, mode='r', encoding='utf-8') as f:
        lines = f.read()

    return parse_dict_str(lines)


def parse_dict_str(s):
    """
    Parse a string as a dict containing nested dicts, lists and Numpy arrays.

    Parameters
    ----------
    s : str
        A string representing nested dicts, lists and Numpy arrays according
        to a particular syntax.

    Returns
    -------
    dict

    Examples
    --------
    Note in the below example, the triple-quotes had to be escaped:

    >>> s = \"""a = {
    ... a1 = 2
    ... a2 = [
    ...     1
    ...     2
    ...     ]
    ... a3 = {
    ...     x = 20
    ...     y = 30
    ...     z = 40
    ...     }
    ... }\"""

    TODO:
    -   Investigate refactoring some of the repeated code in the two main
        loops.

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

    OPEN_OBJ_DEF = {
        '[': [],
        '{': {}
    }

    DTYPES_DEF = {
        'int': int,
        'float': float,
        'bool': bool,
        'str': str
    }

    # Loop through lines twice, first pass to get shape of any arrays,
    # second to parse data.

    arr_shapes = []
    arr_dtypes = []
    cur_arr_consec_blocks = []
    cur_arr_consec_empties = []
    parse_consec_empties = False
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
                dtype_str, v_2 = v.split(DTYPE_OPEN)[1].split(DTYPE_CLOSE)

                if DTYPES_DEF.get(dtype_str, None) is None:
                    dtype_str = None

                else:
                    v = v_2

            if v == ARR_OPEN:
                arr_start = True
                arr_dtypes.append(DTYPES_DEF.get(dtype_str, None))

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
            if len(cur_arr_consec_empties) > 0:

                # Get max number of consecutive blank lines:
                max_blanks = max(cur_arr_consec_empties)
                num_i = 1

                for i in range(max_blanks, 0, -1):

                    if i not in cur_arr_consec_empties:

                        raise ValueError(
                            'Malformed array on line number {}: '
                            'Some blocks are separated by {} blank lines,'
                            'but there are no blocks separated by {} blank '
                            'lines.'.format(ln_idx, max_blanks, i))

                    num_i = 1 + (cur_arr_consec_empties.count(i) / num_i)
                    cur_arr_shape.append(int(num_i))

            cur_arr_shape.extend(cur_arr_consec_blocks[0])
            arr_shapes.append(tuple(cur_arr_shape))

            cur_arr_consec_blocks = []
            cur_arr_consec_empties = []
            parse_consec_blocks = False

        if arr_start:
            arr_start = False
            parse_arr_shape = True
            continue

        if parse_arr_shape:

            if ln == '':
                if not parse_consec_empties:
                    parse_consec_empties = True
                    cur_arr_consec_empties.append(1)

                else:
                    cur_arr_consec_empties[-1] += 1

                if parse_consec_blocks:
                    parse_consec_blocks = False

            else:
                width = len(ln.split())

                if parse_consec_empties:
                    parse_consec_empties = False

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
                                ln_idx + 1, prev_width, width))

    arr_row_idx = [get_consec_arr_row_idx(i) for i in arr_shapes]

    # arr_idx is the index of the arrays whose shapes have been found
    # arr_ln_idx is the index of the row (innermost dimensions) within an array
    arr_idx = -1
    is_arr = False
    arr_ln_idx = -1

    depth_change = 1
    new_is_idx_elm = False
    dtype_str = None
    out = {}
    address = []
    is_idx = []

    # Second loop, to parse data:
    for ln_idx, ln in enumerate(ln_s):

        ln = ln.strip()

        if ln == '':
            continue

        k, v, open_obj, open_str = None, None, None, None

        # Set the line type: ASSIGN_OPEN, ASSIGN, OPEN, CLOSE or VALUE
        if ASSIGN_SYM in ln:

            dtype_str = None

            k, v = [i.strip() for i in ln.split(ASSIGN_SYM)]

            if DTYPE_OPEN in v and DTYPE_CLOSE in v:
                dtype_str, v_2 = v.split(DTYPE_OPEN)[1].split(DTYPE_CLOSE)

                if DTYPES_DEF.get(dtype_str, None) is None:
                    dtype_str = None

                else:
                    v = v_2

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

            if is_arr:
                address = address[:-arr_row_idx[arr_idx].shape[1]]
                is_idx = is_idx[:-arr_row_idx[arr_idx].shape[1]]
                is_arr = False
                arr_ln_idx = -1

            else:
                address = address[:-1]
                is_idx = is_idx[:-1]

        else:

            # Find new address element:
            if ln_type in ['ASSIGN_OPEN', 'ASSIGN']:
                new_address_elem = k

            elif ln_type in ['OPEN', 'VALUE']:
                # set_in_dict appends list element for address element `None`:
                new_address_elem = None

            # Set new address element:
            if depth_change == 0:
                address[-1] = new_address_elem

            elif depth_change == 1 and not is_arr:
                # For an array, we deal with address and is_idx later:
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
                    set_val = parse_string_as(v, DTYPES_DEF[dtype_str])

            elif ln_type == 'VALUE':
                set_val = v

            elif ln_type in ['ASSIGN_OPEN', 'OPEN']:
                set_val = open_obj

            if open_str == ARR_OPEN:

                arr_idx += 1
                set_val = np.zeros(arr_shapes[arr_idx],
                                   dtype=arr_dtypes[arr_idx])
                is_arr = True

            if is_arr:

                if arr_ln_idx >= 0:

                    # Parse an array row
                    set_val = [i for i in set_val.split()]

                    if arr_ln_idx == 0:
                        address += list(arr_row_idx[arr_idx][arr_ln_idx])
                        is_idx += [True, ] * arr_row_idx[arr_idx].shape[1]

                    else:
                        address[-arr_row_idx[arr_idx].shape[1]:] = list(
                            arr_row_idx[arr_idx][arr_ln_idx])
                        is_idx[-arr_row_idx[arr_idx].shape[1]:] = [
                            True, ] * arr_row_idx[arr_idx].shape[1]

                arr_ln_idx += 1

            out = set_in_dict(out, address, is_idx, set_val)

            # For the next line:
            if open_str in [LIST_OPEN, ARR_OPEN]:
                new_is_idx_elm = True
            else:
                new_is_idx_elm = False

    return out
