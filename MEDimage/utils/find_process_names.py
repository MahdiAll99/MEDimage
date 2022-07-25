from inspect import stack, getmodule
from typing import List

def get_process_names() -> List:
    """Get process names

    Returns:
        List: process names
    """
    module_names = ["none"]
    for stack_entry in stack():
        current_module = getmodule(stack_entry[0])
        if current_module is not None:
            module_names += [current_module.__name__]

    return module_names