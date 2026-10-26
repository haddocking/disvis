"""Real 3D FFTs on the GPU with VkFFT, through pyvkfft."""

import numpy as np
from pyvkfft.opencl import VkFFTApp


def ft_shape(shape):
    """Return the shape of the real FFT of an array of the given shape."""
    return tuple(shape[:-1]) + (shape[-1] // 2 + 1,)


class RFFTn:
    """Forward and inverse real 3D FFTs of the given shape on queue.

    The transforms work on pyopencl arrays: float32 arrays of the given
    shape and complex64 arrays of shape ft_shape(shape). As with numpy.fft,
    the inverse transform is scaled by 1/N. The inverse transform may
    overwrite its input.
    """

    def __init__(self, queue, shape):
        self.queue = queue
        self.shape = tuple(shape)
        self.ft_shape = ft_shape(self.shape)
        self._app = VkFFTApp(
            self.shape,
            np.float32,
            queue,
            ndim=len(self.shape),
            inplace=False,
            r2c=True,
            norm=1,
        )

    def rfftn(self, inarray, outarray):
        """Enqueue the forward transform of inarray into outarray."""
        self._app.fft(inarray, outarray, queue=self.queue)

    def irfftn(self, inarray, outarray):
        """Enqueue the inverse transform of inarray into outarray."""
        self._app.ifft(inarray, outarray, queue=self.queue)
