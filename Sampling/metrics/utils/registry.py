_FUNCTION_REGISTRY = {}

def get_registry():
    return _FUNCTION_REGISTRY.copy()
#enddef

def get_function(name: str):
    return _FUNCTION_REGISTRY.get(name)
#enddef