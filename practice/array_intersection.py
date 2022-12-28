def array_int(a, b):
    a, b = sorted(a), sorted(b)
    i_arr = []
    i, j = 0, 0

    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i_arr.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1

    return i_arr


if __name__ == '__main__':
    assert array_int([1, 2, 3], [3, 4, 5]) == [3]
    assert array_int([], [2, 3]) == []