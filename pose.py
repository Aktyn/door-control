import collections
import enum
import math
import os
import time

import cv2
import tensorflow as tf
from pycoral.adapters.common import input_size
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

EDGETPU_SHARED_LIB = 'libedgetpu.so.1'
POSENET_SHARED_LIB = os.path.join(
    'posenet_lib', os.uname().machine, 'posenet_decoder.so')


class KeypointType(enum.IntEnum):
    """Pose keypoints."""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


Point = collections.namedtuple('Point', ['x', 'y'])
Point.distance = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
Point.distance = staticmethod(Point.distance)

Keypoint = collections.namedtuple('Keypoint', ['point', 'score'])

Pose = collections.namedtuple('Pose', ['keypoints', 'score'])

EDGES = (
    (KeypointType.NOSE, KeypointType.LEFT_EYE),
    (KeypointType.NOSE, KeypointType.RIGHT_EYE),
    (KeypointType.NOSE, KeypointType.LEFT_EAR),
    (KeypointType.NOSE, KeypointType.RIGHT_EAR),
    (KeypointType.LEFT_EAR, KeypointType.LEFT_EYE),
    (KeypointType.RIGHT_EAR, KeypointType.RIGHT_EYE),
    (KeypointType.LEFT_EYE, KeypointType.RIGHT_EYE),
    (KeypointType.LEFT_SHOULDER, KeypointType.RIGHT_SHOULDER),
    (KeypointType.LEFT_SHOULDER, KeypointType.LEFT_ELBOW),
    (KeypointType.LEFT_SHOULDER, KeypointType.LEFT_HIP),
    (KeypointType.RIGHT_SHOULDER, KeypointType.RIGHT_ELBOW),
    (KeypointType.RIGHT_SHOULDER, KeypointType.RIGHT_HIP),
    (KeypointType.LEFT_ELBOW, KeypointType.LEFT_WRIST),
    (KeypointType.RIGHT_ELBOW, KeypointType.RIGHT_WRIST),
    (KeypointType.LEFT_HIP, KeypointType.RIGHT_HIP),
    (KeypointType.LEFT_HIP, KeypointType.LEFT_KNEE),
    (KeypointType.RIGHT_HIP, KeypointType.RIGHT_KNEE),
    (KeypointType.LEFT_KNEE, KeypointType.LEFT_ANKLE),
    (KeypointType.RIGHT_KNEE, KeypointType.RIGHT_ANKLE),
)


def main():
    edgetpu_delegate = load_delegate(EDGETPU_SHARED_LIB)
    posenet_decoder_delegate = load_delegate(POSENET_SHARED_LIB)

    # model = './models/posenet_mobilenet_v1_075_353_481_quant_decoder_edgetpu.tflite'  # mobilenet lite
    model = './models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite'  # mobilenet normal
    # model = './models/posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite' # mobilenet large
    # model = './models/posenet_resnet_50_960_736_32_quant_edgetpu_decoder.tflite'  # resnet large

    interpreter = Interpreter(
        model,
        experimental_delegates=[edgetpu_delegate, posenet_decoder_delegate])

    interpreter.allocate_tensors()
    width, height = input_size(interpreter)
    print('width:', width, 'height:', height)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cap = cv2.VideoCapture(0)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    box_x, box_y, box_w, box_h = (0, 0, width, height)
    scale_x, scale_y = frame_width / box_w, frame_height / box_h
    white = (255, 255, 255)
    blue = (255, 0, 0)
    light_green = (128, 255, 128)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    mirror = False
    threshold = 0.05

    def transform_to_source_space(point_in: Point):
        return int((point_in[0] - box_x) * scale_x), int((point_in[1] - box_y) * scale_y)

    counter = 0
    timer = 0
    timer_label = ""
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
            interpreter.set_tensor(input_details[0]['index'], tensor)
            interpreter.invoke()

            [keypoints, keypoint_scores, pose_scores, num_poses] = map(
                lambda details: interpreter.get_tensor(details['index']),
                output_details
            )

            poses = []
            for i in range(int(num_poses[0])):
                pose_score = pose_scores[0][i]
                pose_keypoints = {}
                for j, point in enumerate(keypoints[0][i]):
                    y, x = point
                    if mirror:
                        x = width - x
                    pose_keypoints[KeypointType(j)] = Keypoint(
                        Point(x, y), keypoint_scores[0][i, j])
                poses.append(Pose(pose_keypoints, pose_score))

            xys = {}
            for pose in poses:
                for a, b in EDGES:
                    if a not in pose.keypoints or b not in pose.keypoints:
                        continue
                    ax, ay = transform_to_source_space(pose.keypoints[a].point)
                    bx, by = transform_to_source_space(pose.keypoints[b].point)
                    score_avg = min(1.0, max(0.1, (pose.keypoints[a].score + pose.keypoints[b].score) / 2.0))
                    lines_overlay = cv2_im.copy()
                    cv2.line(lines_overlay, (int(ax), int(ay)), (int(bx), int(by)), light_green, 2)
                    cv2.addWeighted(lines_overlay, score_avg, cv2_im, 1 - score_avg, 0, cv2_im)
                for label, keypoint in pose.keypoints.items():
                    if keypoint.score < threshold:
                        continue
                    kp_x, kp_y = transform_to_source_space(keypoint.point)

                    xys[label] = (kp_x, kp_y, keypoint.score)
                    cv2_im = cv2.circle(cv2_im, (int(kp_x), int(kp_y)), max(1, int(5 * keypoint.score)), blue, -1)

            end_time = time.time()
            duration = (end_time - start_time) * 1000
            counter += 1
            timer += duration
            if counter >= 30:
                timer_label = f"{round(timer / counter)}ms"
                counter = 0
                timer = 0

            cv2_im = cv2.putText(cv2_im, timer_label, (5, 15), font, font_scale, white, 1)
            cv2.imshow('Real time pose estimation', cv2_im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)

    print("Done")


if __name__ == '__main__':
    main()
