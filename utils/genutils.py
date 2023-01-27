

def listify(inp, none_to='none', set_to_list=False):

    if inp is None:
        if none_to == 'none':
            return None
        elif none_to == 'empty':
            return []
        elif none_to == 'list_of_none':
            return [None]
        else:
            raise ValueError("Incorrect input for none_to")
    elif isinstance(inp, list):
        return inp
    elif isinstance(inp, set):
        return list(inp) if set_to_list else inp

