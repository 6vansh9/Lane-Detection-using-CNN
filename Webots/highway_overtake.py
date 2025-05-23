# Copyright 1996-2023 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vehicle_driver controller."""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from vehicle import Driver

lane_detection_model = load_model('D:/Projects/ADAS/full_CNN_modell.h5')
sensorsNames = [
    "front",
    "front right 0",
    "front right 1",
    "front right 2",
    "front left 0",
    "front left 1",
    "front left 2",
    "rear",
    "rear left",
    "rear right",
    "right",
    "left"]
sensors = {}

# Constants for realistic driving behavior
MAX_SPEED = 120  # km/h
MIN_SPEED = 60   # km/h
SAFE_DISTANCE = 20.0  # meters
OVERTAKE_DISTANCE = 30.0  # meters
ACCELERATION_RATE = 2.0  # km/h per step
DECELERATION_RATE = 3.0  # km/h per step
STEERING_SMOOTHING = 0.3  # Smoothing factor for steering

lanePositions = [10.6, 6.875, 3.2]
currentLane = 1
overtakingSide = None
targetSpeed = MAX_SPEED
currentSpeed = 0
safeOvertake = True
overtakeTimer = 0
MAX_OVERTAKE_TIME = 100  # Maximum time to complete overtaking maneuver


def apply_PID(position, targetPosition):
    p_coefficient = 0.05
    i_coefficient = 0.000015
    d_coefficient = 25
    diff = position - targetPosition
    if apply_PID.previousDiff is None:
        apply_PID.previousDiff = diff
    # anti-windup mechanism
    if diff > 0 and apply_PID.previousDiff < 0:
        apply_PID.integral = 0
    if diff < 0 and apply_PID.previousDiff > 0:
        apply_PID.integral = 0
    apply_PID.integral += diff
    # compute angle
    angle = p_coefficient * diff + i_coefficient * apply_PID.integral + d_coefficient * (diff - apply_PID.previousDiff)
    apply_PID.previousDiff = diff
    return angle


apply_PID.integral = 0
apply_PID.previousDiff = None


def get_filtered_speed(speed):
    get_filtered_speed.previousSpeeds.append(speed)
    if len(get_filtered_speed.previousSpeeds) > 100:  
        get_filtered_speed.previousSpeeds.pop(0)
    return sum(get_filtered_speed.previousSpeeds) / float(len(get_filtered_speed.previousSpeeds))


def is_vehicle_on_side(side):
    for i in range(3):
        name = "front " + side + " " + str(i)
        if sensors[name].getValue() > 0.8 * sensors[name].getMaxValue():
            return True
    return False


def reduce_speed_if_vehicle_on_side(speed, side):
    minRatio = 1
    for i in range(3):
        name = "front " + overtakingSide + " " + str(i)
        ratio = sensors[name].getValue() / sensors[name].getMaxValue()
        if ratio < minRatio:
            minRatio = ratio
    return minRatio * speed


get_filtered_speed.previousSpeeds = []
driver = Driver()
for name in sensorsNames:
    sensors[name] = driver.getDevice("distance sensor " + name)
    sensors[name].enable(10)

gps = driver.getDevice("gps")
gps.enable(10)

camera = driver.getDevice("camera")
camera.enable(10)
camera.recognitionEnable(50)

def apply_lane_detection(image):
    small_img = cv2.resize(image[:, :, :3], (160, 80))
    small_img = np.array(small_img)
    small_img = small_img[None, :, :, :]

    prediction = lane_detection_model.predict(small_img)[0] * 255

    prediction = prediction.astype(np.uint8)

    return prediction

def calculate_safe_speed(front_distance, front_range):
    """Calculate safe speed based on distance to vehicle ahead"""
    if front_distance < SAFE_DISTANCE:
        return MIN_SPEED
    elif front_distance < OVERTAKE_DISTANCE:
        return MIN_SPEED + (MAX_SPEED - MIN_SPEED) * (front_distance - SAFE_DISTANCE) / (OVERTAKE_DISTANCE - SAFE_DISTANCE)
    return MAX_SPEED

def smooth_steering(current_angle, target_angle):
    """Apply smoothing to steering angle changes"""
    return current_angle + (target_angle - current_angle) * STEERING_SMOOTHING

def is_safe_to_overtake(side):
    """Check if it's safe to overtake on the specified side"""
    if side == 'left':
        # Check left side sensors
        if (sensors["left"].getValue() < 0.8 * sensors["left"].getMaxValue() or
            sensors["rear left"].getValue() < 0.8 * sensors["rear left"].getMaxValue()):
            return False
    else:
        # Check right side sensors
        if (sensors["right"].getValue() < 0.8 * sensors["right"].getMaxValue() or
            sensors["rear right"].getValue() < 0.8 * sensors["rear right"].getMaxValue()):
            return False
    return True

while driver.step() != -1:
    frontDistance = sensors["front"].getValue()
    frontRange = sensors["front"].getMaxValue()
    
    # Calculate target speed based on safety
    targetSpeed = calculate_safe_speed(frontDistance, frontRange)
    
    # Smooth speed changes
    if currentSpeed < targetSpeed:
        currentSpeed = min(currentSpeed + ACCELERATION_RATE, targetSpeed)
    else:
        currentSpeed = max(currentSpeed - DECELERATION_RATE, targetSpeed)
    
    # Handle overtaking logic
    if overtakingSide is not None:
        overtakeTimer += 1
        if overtakeTimer > MAX_OVERTAKE_TIME:
            # Abort overtaking if taking too long
            overtakingSide = None
            overtakeTimer = 0
        elif not is_safe_to_overtake(overtakingSide):
            # Abort if conditions become unsafe
            overtakingSide = None
            overtakeTimer = 0
        else:
            # Adjust speed during overtaking
            currentSpeed = min(currentSpeed, MAX_SPEED * 0.9)
    
    # Apply speed control
    driver.setCruisingSpeed(currentSpeed)
    speedDiff = driver.getCurrentSpeed() - currentSpeed
    if speedDiff > 0:
        driver.setBrakeIntensity(min(speedDiff / currentSpeed, 1))
    else:
        driver.setBrakeIntensity(0)
    
    # Overtaking decision making
    if frontDistance < 0.8 * frontRange and overtakingSide is None:
        if (is_vehicle_on_side("left") and
                is_safe_to_overtake("left") and
                currentLane < 2):
            currentLane += 1
            overtakingSide = 'right'
            overtakeTimer = 0
        elif (is_vehicle_on_side("right") and
                is_safe_to_overtake("right") and
                currentLane > 0):
            currentLane -= 1
            overtakingSide = 'left'
            overtakeTimer = 0
    
    # Lane position control with smoothing
    position = gps.getValues()[1]
    targetAngle = apply_PID(position, lanePositions[currentLane])
    currentAngle = driver.getSteeringAngle()
    smoothedAngle = smooth_steering(currentAngle, targetAngle)
    driver.setSteeringAngle(-smoothedAngle)
    
    # Reset overtaking state when lane change is complete
    if abs(position - lanePositions[currentLane]) < 1.0:
        overtakingSide = None
        overtakeTimer = 0
    
    # Process camera image and lane detection
    image = camera.getImage()
    image = np.frombuffer(image, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    lane_detection_result = apply_lane_detection(image)
    
    # Display results
    cv2.imshow("Lane Detection Result", lane_detection_result)
    cv2.waitKey(1)
