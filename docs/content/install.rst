Install
=======

PyPI
-----
::

  pip install pymltoolkit
  
If the installation failed with dependancy issues, execute the above command with --no-dependencies ::

  pip install pymltoolkit --no-dependencies
  
Setup TensorFlow with GPU support (Optional)
--------------------------------------------
Refer the official TensorFlow documentation (https://www.tensorflow.org/install/gpu) for most up to date innstructions.

PyMLToolKit is tested with the following software versions in Windows 10
  * CUDA Toolkit 10.0 (10.0.130_411.31_win10)
  * cuDNN SDK (v7.4.2.24)

Step #1
  * Install latest NVIDIA® GPU drivers
  * Install CUDA Toolkit
  * Install cuDNN SDK
  * Set System Path to CUDA Toolkit. If the CUDA Toolkit is installed to "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0" and extracted cuDNN content to r"C:/Program Files\NVIDIA GPU Computing Toolkit/cuDNN", update your %PATH% to match:
  ::
  
    SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin;%PATH%
    SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\extras\CUPTI\libx64;%PATH%
    SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include;%PATH%
    SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\cuDNN\bin;%PATH%

Step #2
  * Install TensorFLow-GPU
  
  PyPI
  ::
    pip install tensorflow-gpu
    
  To install specific version
  ::
    pip install tensorflow-gpu==1.14
    
  * Check GPU in Tensorflow (output forat as below)
  ::
  
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    
  ::
  
    [name: "/device:CPU:0"
    device_type: "CPU"
    memory_limit: 99999999
    locality {
    }
    incarnation: 9999999999, name: "/device:GPU:0"
    device_type: "GPU"
    memory_limit: 99999999
    locality {
       bus_id: 1
       links {
       }
    }
    incarnation: 99999999
    physical_device_desc: "device: 0, name: XXXXXX, pci bus id: 0000:00:00.0, compute capability: 0.0"]
    
memory_limit is in bytes. To convert allocated memeory to GB use : memory_limit/(1024*1024*1024)

If you encounter errors in setting up TensorFlow, please refer to thw official TensorFlow Build and install error messages (https://www.tensorflow.org/install/errors)

