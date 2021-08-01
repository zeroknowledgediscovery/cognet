def assert_None(args,
                raise_error=True):
    '''Make sure args are not None
    '''
    if any(x is None for x in args):
        num_none = sum(x is None for x in args)
        if raise_error:
            string='Nones detected : {}'.format(str(num_none))
            raise ValueError(string)
        else:
            return num_none

def assert_array_dimension(array, dimensions):
    '''
    '''
    if len(array.shape) != dimensions:
        raise ValueError('You must pass in a {}-D array!'.format(dimensions))
        
#print(assert_None([None], False))