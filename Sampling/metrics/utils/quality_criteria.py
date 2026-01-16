from enum import Enum
from .registry import _FUNCTION_REGISTRY, get_registry, get_function


class QualityCriteria(Enum):
    COMPLEXITY = 'complexity'
    SIMPLICITY = 'simplicity'
    GENERALIZATION = 'generalization'
    FITNESS = 'fitness'
    PRECISION = 'precision'

    def __str__(self):
        return self.value
    #enddef
#endclass

def get_function_by_quality_criteria(name: str):
    func = get_function(name)
    if func:
        return getattr(func, '_quality_criteria')
    return None
#enddef

def find_functions_by_quality_criteria(desired_criteria: QualityCriteria):
    all_functions = get_registry()
    return [
        func for func in all_functions.values()
        if getattr(func, '_quality_criteria', None) == desired_criteria
    ]
#enddef

def classify(criteria_enum: QualityCriteria):
    def decorator(func):
        func._quality_criteria = criteria_enum
        _FUNCTION_REGISTRY[func.__name__] = func
        return func
    #enddef
    return decorator
#enddef