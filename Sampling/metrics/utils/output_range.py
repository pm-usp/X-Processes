from enum import Enum
from .registry import _FUNCTION_REGISTRY, get_registry, get_function


class SymbolicRange(Enum):
    NEG_N_TO_ZERO_EXCLUSIVE = '(-n, 0)'
    NEG_N_TO_ONE_EXCLUSIVE = '(-n, 1)'
    NEG_N_TO_ONE_RIGHT_INCLUSIVE = '(-n, 1]'
    NEG_N_TO_N_EXCLUSIVE = '(-n, n)'
    ZERO_TO_N_EXCLUSIVE = '[0, n)'
    ONE_TO_N_EXCLUSIVE = '[1, n)'
    ZERO_TO_N_TUPLE = '([0, n), [0, n))'
    ZERO_TO_ONE_LEFT_EXCLUSIVE = '(0, 1]'
    NEG_ONE_TO_ONE_INCLUSIVE = '[-1, 1]'
    ZERO_TO_ONE_RIGHT_EXCLUSIVE = '[0, 1)'
    ZERO_TO_ONE_INCLUSIVE = '[0, 1]'

    def __str__(self):
        return self.value
    #enddef
#endclass

def get_function_output_range(name: str):
    func = get_function(name)
    if func:
        return getattr(func, '_output_range')
    return None
#enddef

def find_functions_by_range(desired_range: SymbolicRange):
    all_functions = get_registry()
    return [
        func for func in all_functions.values()
        if getattr(func, '_output_range') == desired_range
    ]
#enddef

def output_range(range_enum: SymbolicRange):
    def decorator(func):
        func._output_range = range_enum
        _FUNCTION_REGISTRY[func.__name__] = func
        return func
    #enddef
    return decorator
#enddef