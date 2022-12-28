class Node:
    def __int__(self, val, left=None, right=None):
        """ Constructor """
        self.val = val
        self.right = right
        self.left = left


def tree_depth(root: Node):
    """ Finds tree depth """
    if root is None:
        return 0
    else:
        return max(tree_depth(root.left), tree_depth(root.right)) + 1


def tree_diameter(root: Node):
    if root is None:
        return 0
    else:
        return tree_depth(root.left) + tree_depth(root.right)

