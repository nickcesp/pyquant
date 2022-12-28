from math import sqrt
import heapq


def k_closest_points(points, k):

    if len(points) < k:
        raise ValueError("k must be at most the length of  points")

    # Calculate the distance to origin
    d = [sqrt(p[0] ** 2 + p[1] ** 2) for p in points]
    z = zip(d, points)
    return [t[1] for t in heapq.nsmallest(k, z)]


if __name__ == '__main__':
    r = k_closest_points([[2, -1], [3, 2], [4, 1], [-1, -1], [-2, 2]], 3)
    print(r)