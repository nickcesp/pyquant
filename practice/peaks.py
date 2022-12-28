import numpy as np


def all_peak_elements(arr):
    last_inc = True
    d_arr = np.diff(arr)
    ret = []
    for i in range(len(d_arr)):
        if not last_inc:
            last_inc = d_arr[i] > 0
        elif d_arr[i] < 0:
            last_inc = False
            ret.append(i)
    if d_arr[-1] > 0:
        ret.append(len(d_arr))

    return ret

def pe_bs(arr):
    start = 0
    end = len(arr) - 1

    while True:
        mid = (start + end) // 2
        left = arr[mid-1] if mid - 1 >= 0 else float("-inf")
        right = arr[mid+1] if mid + 1 < len(arr) else float("-inf")

        if left < arr[mid] and right < arr[mid]:
            return mid
        elif arr[mid] < right:
            start = mid + 1
        else:
            end = mid - 1


if __name__ == '__main__':
    assert all_peak_elements([1, 2, 3, 0, 3, -2]) == [2, 4]
    assert all_peak_elements(list(range(10))) == [9]
    assert all_peak_elements(list(range(10, -1, -1))) == [0]
    assert all_peak_elements([2, 0, 1, 2, 3]) == [0, 4]
