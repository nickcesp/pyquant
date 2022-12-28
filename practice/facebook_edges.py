import queue

def min_path(n, edges, x, y):
    """ Computes the min path between 2 users using breadth first search

        :param n: Number of nodes in edge
        :param edges: 2d array 

    """
    dist = [0] * n
    processed = [False] * n
    q = queue.Queue()
    q.put(x)
    while not q.empty():
        curr = q.get()



if __name__ == '__main__':
