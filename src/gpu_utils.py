#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch

def print_cuda_info():
    """
    Print basic CUDA device information if available.
    """
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        dev_id = torch.cuda.current_device()
        print("Current device index:", dev_id)
        print("Device name:", torch.cuda.get_device_name(dev_id))
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"GPU memory [MB] - total: {info.total // 1024**2}, used: {info.used // 1024**2}, free: {info.free // 1024**2}")
        except Exception:
            print("Optional: install pynvml for detailed memory info (pip install pynvml).")

