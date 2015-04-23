import os
import errno
try:
    import pyopencl as cl
except ImportError:
    pass
    

def get_queue():
    try:
        platform = cl.get_platforms()[0]
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        context = cl.Context(devices=devices)
        queue = cl.CommandQueue(context, device=devices[0])
    except:
        queue = None

    return queue


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def parse_interactions(f):
    data = {}
    with open(f) as f:
        words = f.readline().split()
        data['residues'] = [int(x) for x in words[2:]]

        for line in f:
            words = line.split()

            consistent_restraints = int(words[0])
            data[consistent_restraints] = {}

            data[consistent_restraints]['total'] = int(words[1])
            data[consistent_restraints]['interactions'] = [int(x) for x in words[2:]]
    return data

