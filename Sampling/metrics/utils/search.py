from .registry import get_registry
from .output_range import SymbolicRange
from .quality_criteria import QualityCriteria

def find_functions(quality_criteria: QualityCriteria = None, func_range: SymbolicRange = None):
    """
    Finds functions by any combination of criteria (category, range, etc.).
    """

    # Start with all registered functions
    candidates = list(get_registry().values())

    # Filter by category, if provided
    if quality_criteria:
        candidates = [
            func for func in candidates
            if getattr(func, '_quality_criteria', None) == quality_criteria
        ]
    #endif

    # Filter the remaining candidates by range, if provided
    if func_range:
        candidates = [
            func for func in candidates
            if getattr(func, '_output_range', None) == func_range
        ]
    #endif

    return candidates
#enddef