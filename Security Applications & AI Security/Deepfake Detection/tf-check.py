import tensorflow as tf
from tensorflow.python.client import device_lib

tf.test.is_built_with_cuda()
tf.config.list_physical_devices('GPU')
tf.test.is_gpu_available('GPU') #위의 함수로 바뀐다고 함 (2.4.0)
tf.sysconfig.get_build_info()

print(device_lib.list_local_devices())


