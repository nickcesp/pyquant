class Node:
    def __int__(self, val, left=None, right=None):
        """ Constructor """
        self.val = val
        self.right = right
        self.left = left


class Tree:

    @staticmethod
    def is_mirror(t1: Node, t2: Node):
        """ Checks if t1 is mirror image of t2 """
        if t1 is not None and t2 is not None:
            if t1.val != t2.val:
                return False
            else:
                return Tree.is_mirror(t1.left, t2.right) and Tree.is_mirror(t1.right, t2.left)
        else:
            return t1 is None and t2 is None