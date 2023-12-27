import RPi.GPIO as GPIO
from enum import Enum


class Direction(Enum):
    FORWARD = 1
    BACKWARD = 2


class MotorControl:
    def __init__(self):
        self.in1 = 24
        self.in2 = 23
        en = 25

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.in1, GPIO.OUT)
        GPIO.setup(self.in2, GPIO.OUT)
        GPIO.setup(en, GPIO.OUT)
        GPIO.output(self.in1, GPIO.LOW)
        GPIO.output(self.in2, GPIO.LOW)
        self.p = GPIO.PWM(en, 1000)
        self.p.ChangeDutyCycle(0)

    def __del__(self):
        GPIO.cleanup()

    def run(self, direction: Direction):
        print(f"Running motor in direction {direction}")
        if direction == Direction.FORWARD:
            GPIO.output(self.in1, GPIO.HIGH)
            GPIO.output(self.in2, GPIO.LOW)
        elif direction == Direction.BACKWARD:
            GPIO.output(self.in1, GPIO.LOW)
            GPIO.output(self.in2, GPIO.HIGH)

        self.p.start(100)

    def stop(self):
        print("Stopping motor")
        GPIO.output(self.in1, GPIO.LOW)
        GPIO.output(self.in2, GPIO.LOW)
