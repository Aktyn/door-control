import time
import cv2
import numpy as np
import tensorflow as tf
from pycoral.adapters.common import input_size
from pycoral.utils.edgetpu import make_interpreter


def main():
    edge_interpreter = make_interpreter('./models/fast_depth_256x320_edgetpu.tflite')
    # edge_interpreter = make_interpreter('./models/fast_depth_480x640_edgetpu.tflite')
    edge_interpreter.allocate_tensors()
    width, height = input_size(edge_interpreter)
    print('width:', width, 'height:', height)

    input_details = edge_interpreter.get_input_details()
    output_details = edge_interpreter.get_output_details()

    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break
            cv2_im = frame

            img = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

            reshaped_img = resized_img.reshape(1, height, width, 3)
            tensor = tf.convert_to_tensor(reshaped_img, dtype=tf.uint8)

            # inference
            edge_interpreter.set_tensor(input_details[0]['index'], tensor)
            edge_interpreter.invoke()
            output = edge_interpreter.get_tensor(output_details[0]['index'])

            tensor = tf.convert_to_tensor(output, dtype=tf.uint8)
            numpy_array = tensor.numpy()
            prediction = cv2.cvtColor(np.float32(numpy_array[0]), cv2.COLOR_RGB2BGR)

            depth_min = prediction.min()
            depth_max = prediction.max()
            img_out = (255 - 255 * (prediction - depth_min) / (depth_max - depth_min)).astype("uint8")
            depth_image = cv2.applyColorMap(img_out, cv2.COLORMAP_JET)

            cv2.imshow('depth', depth_image)
            cv2.imshow('camera', cv2_im)

            end_time = time.time()
            duration = (end_time - start_time) * 1000
            print(f"{round(duration)} milliseconds")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)

    print("Done")


if __name__ == '__main__':
    main()
