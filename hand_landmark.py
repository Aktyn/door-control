import time

import cv2
import tensorflow as tf
from pycoral.adapters.common import input_size
from pycoral.utils.edgetpu import make_interpreter


# TODO: detect hand bounding box and use the cropped image as input to the hand_landmark model


def main():
    landmark_interpreter = make_interpreter('./models/hand_landmark_3d_224.tflite')
    landmark_interpreter.allocate_tensors()
    width, height = input_size(landmark_interpreter)
    print('width:', width, 'height:', height)

    input_details = landmark_interpreter.get_input_details()
    output_details = landmark_interpreter.get_output_details()

    cap = cv2.VideoCapture(0)
    # frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    blue = (255, 0, 0)
    light_green = (128, 255, 128)

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20),
        (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
    ]

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
            landmark_interpreter.set_tensor(input_details[0]['index'], tensor)
            landmark_interpreter.invoke()

            [_, _, confidence, landmarks] = map(
                lambda details: landmark_interpreter.get_tensor(details['index']),
                output_details
            )

            # Draw landmarks as points
            if confidence[0] / 128 > 0.25:
                for connection in connections:
                    x0 = landmarks[0][connection[0] * 3 + 0]
                    y0 = landmarks[0][connection[0] * 3 + 1]
                    x1 = landmarks[0][connection[1] * 3 + 0]
                    y1 = landmarks[0][connection[1] * 3 + 1]
                    cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), light_green, 2)
                for i in range(0, len(landmarks[0]), 3):
                    [x, y, z] = landmarks[0][i:i + 3]
                    size = 1 - z / 128.0
                    cv2_im = cv2.circle(cv2_im, (int(x), int(y)), max(1, round(3 * size)), blue, -1)

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
