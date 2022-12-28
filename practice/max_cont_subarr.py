def max_cont_sub(arr) -> int:
    """ Returns the max sum in any contiguous sub-array within arr
        O(n)
    """
    s, max_seen = 0, 0
    for v in arr:
        if v < 0:
            if s > max_seen:
                max_seen = s
            s = 0
        else:
            s += v

    return max(s, max_seen)

if __name__ == '__main__':
    assert 11 == max_cont_sub([-1, -3, 5, -4, 3, -6, 9, 2])
    assert 16 == max_cont_sub([-1, -3, 5, -4, 3, -6, 9, 2, 5])
    assert 20 == max_cont_sub([-1, -3, 5, -4, 20, -1, 3, -6, 9, 2, 5])
    assert 0 == max_cont_sub([-1, -3, -5, -4, -20, -1, -3, -6, -9, -2, -5])


