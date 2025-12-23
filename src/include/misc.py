import os
import sys
import torch
from contextlib import contextmanager

@contextmanager
def suppress_output():
    """
    Context manager that redirects stdout and stderr to /dev/null.
    Useful for silencing C++ level warnings from torch.compile/max-autotune.
    """
    # Open the null device
    with open(os.devnull, "w") as devnull:
        # 1. Flush the streams to ensure prior output isn't lost or mixed up
        sys.stdout.flush()
        sys.stderr.flush()

        # 2. Save the original file descriptors
        old_stdout_fd = os.dup(sys.stdout.fileno())
        old_stderr_fd = os.dup(sys.stderr.fileno())

        try:
            # 3. Replace stdout/stderr with the null device
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            
            yield
            
        finally:
            # 4. Flush again before restoring
            sys.stdout.flush()
            sys.stderr.flush()
            
            # 5. Restore the original file descriptors
            os.dup2(old_stdout_fd, sys.stdout.fileno())
            os.dup2(old_stderr_fd, sys.stderr.fileno())
            
            # 6. Close the duplicate descriptors
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

def compile_if_needed(module, compile_mode):
    if compile_mode is None:
        return module
    else:
        # Suppress compile output and logs
        with suppress_output():
            return torch.compile(module, dynamic=False, fullgraph=True)