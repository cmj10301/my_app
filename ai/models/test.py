import tensorflow as tf

print("TensorFlow 버전:", tf.__version__)
print("GPU 디바이스 목록:")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print("✅", gpu)
else:
    print("❌ GPU를 찾을 수 없습니다. (CPU만 사용 중)")
