#!/usr/bin/env python
import io
import webbrowser

with open('cpu_nms.pyx', 'r') as file:
    change_line = file.readlines()


change_line[24] = "    cdef np.ndarray[np.int64_t, ndim=1] order = scores.argsort()[::-1].\n"

with open('setup.py', 'w') as file:
    file.writelines(change_line)