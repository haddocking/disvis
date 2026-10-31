"""Unit tests of DisVis."""

from disvis.disvis import PYOPENCL


def opencl_queue():
    """Return an OpenCL queue for the GPU tests, or None to skip them.

    A GPU if available, otherwise any OpenCL device (e.g. pocl on the CPU).
    None without pyopencl and pyvkfft, without an OpenCL device, or when the
    device cannot build a kernel: the pocl 3.0 of PyPI fails on CPUs that its
    LLVM does not know, such as the AMD EPYC 9V45 of some GitHub runners.
    """
    if not PYOPENCL:
        return None
    import pyopencl as cl

    from disvis.helpers import get_queue

    queue = get_queue() or get_queue('all')
    if queue is None:
        return None
    try:
        cl.Program(queue.context, '__kernel void probe() {}').build()
    except cl.Error:
        return None
    return queue
