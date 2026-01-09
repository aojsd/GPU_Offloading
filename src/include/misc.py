import os
import sys
import torch
from contextlib import contextmanager

@contextmanager
def suppress_all_output():
    """
    Redirects file descriptors 1 (stdout) and 2 (stderr) to /dev/null.
    This suppresses output from C++ extensions, Triton, and the Python interpreter.
    """
    # Open a pair of null files
    null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
    
    # Save the actual stdout (1) and stderr (2) file descriptors.
    save_fds = [os.dup(1), os.dup(2)]

    try:
        # Assign the null pointers to stdout and stderr.
        os.dup2(null_fds[0], 1)
        os.dup2(null_fds[1], 2)
        yield
    finally:
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(save_fds[0], 1)
        os.dup2(save_fds[1], 2)
        
        # Close the null files and the saved copies
        for fd in null_fds + save_fds:
            os.close(fd)

def compile_if_needed(module, compile_mode):
    if compile_mode is None:
        return module
    else:
        # Suppress compile output and logs
        with suppress_all_output():
            return torch.compile(module, mode=compile_mode, fullgraph=True)