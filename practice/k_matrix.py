import heapq

def find_diag(k):
    """ return diag number """
    if k <= 0:
        raise ValueError()

    i = 1
    prev_val = 1
    while True:
        if k < i + prev_val:
            return i, k - prev_val
        prev_val += i
        i += 1

def smallest(mat, k):
    """ find kth smallest element in sorted matrix k n x n"""
    d, d_k = find_diag(k)
    print(d, d_k)

    diag_idx = zip(range(min(d - 1, len(m) - 1), -1, -1), range(max(0, d - len(m)), len(m)))
    print(list(zip(range(min(d - 1, len(m) - 1), -1, -1), range(max(0, d - len(m)), len(m)))))
    d_vals = [mat[i][j] for i, j in diag_idx]
    return heapq.nsmallest(d_k + 1, d_vals)[d_k]






if __name__ == '__main__':
    m = [[1, 4, 7], [3, 5, 9], [6, 8, 11]]
    assert smallest(m, 4) == 5
    assert smallest(m, 1) == 1
    assert smallest(m, 3) == 4
    assert smallest(m, 5) == 6
    assert smallest(m, 7) == 8
    assert smallest(m, 8) == 9
    assert smallest(m, 9) == 11
