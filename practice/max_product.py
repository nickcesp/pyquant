import heapq

def max_prod_3(arr):
    """ Returns the max product of any 3 integers in the array """
    a = heapq.nlargest(3, arr)
    b = heapq.nsmallest(2, arr)

    return max(a[0] * a[1] * a[2], a[0] * b[1] * b[2])

