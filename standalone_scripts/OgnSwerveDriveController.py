# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
from isaacsim.core.nodes import BaseResetNode
from isaacsim.core.utils.types import ArticulationAction


class SwerveDriveController:
    """
    Swerve drive controller for robots with independently steerable wheels.
    
    Each wheel module can steer (rotate around vertical axis) and drive independently,
    allowing for omnidirectional movement and zero-radius turning.
    """
    
    def __init__(
        self,
        name: str,
        wheel_radius: np.ndarray,
        wheel_positions: np.ndarray,
        wheel_base_length: float = 1.0,
        wheel_base_width: float = 1.0,
        up_axis: np.ndarray = np.array([0, 0, 1]),
        max_linear_speed: float = 1.0e20,
        max_angular_speed: float = 1.0e20,
        max_wheel_speed: float = 1.0e20,
        max_steering_speed: float = 10.0,
        linear_gain: float = 1.0,
        angular_gain: float = 1.0,
        optimize_steering: bool = True,
    ):
        """
        Args:
            name: Name of the controller
            wheel_radius: Array of wheel radii in meters
            wheel_positions: Array of wheel positions [x, y, z] relative to robot center
            wheel_base_length: Distance between front and rear axles
            wheel_base_width: Distance between left and right wheels
            up_axis: Up direction vector
            max_linear_speed: Maximum linear velocity (m/s)
            max_angular_speed: Maximum angular velocity (rad/s)
            max_wheel_speed: Maximum wheel rotation speed (rad/s)
            max_steering_speed: Maximum steering angular velocity (rad/s)
            linear_gain: Gain for linear velocity
            angular_gain: Gain for angular velocity
            optimize_steering: If True, optimize steering to minimize rotation
        """
        self._name = name
        self._wheel_radius = np.asarray(wheel_radius)
        self._wheel_positions = np.asarray(wheel_positions)
        self._wheel_base_length = wheel_base_length
        self._wheel_base_width = wheel_base_width
        self._up_axis = np.asarray(up_axis)
        self._max_linear_speed = max_linear_speed
        self._max_angular_speed = max_angular_speed
        self._max_wheel_speed = max_wheel_speed
        self._max_steering_speed = max_steering_speed
        self._linear_gain = linear_gain
        self._angular_gain = angular_gain
        self._optimize_steering = optimize_steering
        
        self._num_wheels = len(wheel_radius)
        self._previous_steering_angles = np.zeros(self._num_wheels)
        
    def forward(self, command: np.ndarray) -> tuple:
        """
        Compute wheel drive velocities and steering angles from chassis velocity command.
        
        Args:
            command: [vx, vy, omega] - linear velocities (m/s) and angular velocity (rad/s)
            
        Returns:
            tuple: (drive_velocities, steering_angles)
                - drive_velocities: wheel angular velocities in rad/s
                - steering_angles: steering angles in radians
        """
        # Extract and scale command
        vx = command[0] * self._linear_gain
        vy = command[1] * self._linear_gain
        omega = command[2] * self._angular_gain
        
        # Apply velocity limits
        linear_speed = np.sqrt(vx**2 + vy**2)
        if linear_speed > self._max_linear_speed:
            scale = self._max_linear_speed / linear_speed
            vx *= scale
            vy *= scale
            
        if abs(omega) > self._max_angular_speed:
            omega = np.sign(omega) * self._max_angular_speed
        
        # Initialize output arrays
        drive_velocities = np.zeros(self._num_wheels)
        steering_angles = np.zeros(self._num_wheels)
        
        # Calculate velocity and steering angle for each wheel
        for i in range(self._num_wheels):
            # Get wheel position (x, y relative to robot center)
            wheel_x = self._wheel_positions[i][0]
            wheel_y = self._wheel_positions[i][1]
            
            # Calculate wheel velocity components
            # Linear contribution + rotational contribution
            wheel_vx = vx - omega * wheel_y
            wheel_vy = vy + omega * wheel_x
            
            # Calculate wheel speed and angle
            wheel_speed = np.sqrt(wheel_vx**2 + wheel_vy**2)
            wheel_angle = np.arctan2(wheel_vy, wheel_vx)
            
            # Optimize steering angle to minimize rotation
            if self._optimize_steering:
                angle_diff = self._normalize_angle(wheel_angle - self._previous_steering_angles[i])
                
                # If angle difference is more than 90 degrees, flip the wheel
                if abs(angle_diff) > np.pi / 2:
                    wheel_angle = self._normalize_angle(wheel_angle + np.pi)
                    wheel_speed = -wheel_speed
            
            # Apply wheel speed limit
            wheel_angular_velocity = wheel_speed / self._wheel_radius[i]
            if abs(wheel_angular_velocity) > self._max_wheel_speed:
                wheel_angular_velocity = np.sign(wheel_angular_velocity) * self._max_wheel_speed
            
            drive_velocities[i] = wheel_angular_velocity
            steering_angles[i] = wheel_angle
            
        # Update previous steering angles
        self._previous_steering_angles = steering_angles.copy()
        
        return drive_velocities, steering_angles
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        # while angle > np.pi:
        #     angle -= 2 * np.pi
        # while angle < -np.pi:
        #     angle += 2 * np.pi
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        return angle


class OgnSwerveDriveControllerInternalState(BaseResetNode):
    def __init__(self):
        self.wheel_radius = [0.0]
        self.wheel_positions = np.array([])
        self.wheel_base_length = 1.0
        self.wheel_base_width = 1.0
        self.up_axis = np.array([0, 0, 1])
        self.controller_handle = None
        self.max_linear_speed = 1.0e20
        self.max_angular_speed = 1.0e20
        self.max_wheel_speed = 1.0e20
        self.max_steering_speed = 10.0
        self.linear_gain = 1.0
        self.angular_gain = 1.0
        self.optimize_steering = True
        self.node = None
        self.graph_id = None
        super().__init__(initialize=False)

    def initialize_controller(self) -> None:
        self.controller_handle = SwerveDriveController(
            name="swerve_drive_controller",
            wheel_radius=np.asarray(self.wheel_radius),
            wheel_positions=np.asarray(self.wheel_positions),
            wheel_base_length=self.wheel_base_length,
            wheel_base_width=self.wheel_base_width,
            up_axis=self.up_axis,
            max_linear_speed=self.max_linear_speed,
            max_angular_speed=self.max_angular_speed,
            max_wheel_speed=self.max_wheel_speed,
            max_steering_speed=self.max_steering_speed,
            linear_gain=self.linear_gain,
            angular_gain=self.angular_gain,
            optimize_steering=self.optimize_steering,
        )
        self.initialized = True

    def forward(self, command: np.ndarray) -> tuple:
        return self.controller_handle.forward(command)

    def custom_reset(self):
        if self.initialized:
            self.node.get_attribute("inputs:inputVelocity").set([0, 0, 0])
            self.node.get_attribute("outputs:driveVelocityCommand").set([])
            self.node.get_attribute("outputs:steeringAngleCommand").set([])
            # Reset previous steering angles
            self.controller_handle._previous_steering_angles = np.zeros(len(self.wheel_radius))


class OgnSwerveDriveController:
    """
    OmniGraph node for controlling swerve drive robots with independent wheel steering and drive.
    """

    @staticmethod
    def init_instance(node, graph_instance_id):
        state = OgnSwerveDriveControllerDatabase.get_internal_state(node, graph_instance_id)
        state.node = node
        state.graph_id = graph_instance_id

    @staticmethod
    def release_instance(node, graph_instance_id):
        try:
            state = OgnSwerveDriveControllerDatabase.get_internal_state(node, graph_instance_id)
        except Exception:
            state = None
            pass

        if state is not None:
            state.reset()
            state.initialized = False

    @staticmethod
    def internal_state():
        return OgnSwerveDriveControllerInternalState()

    @staticmethod
    def compute(db) -> bool:
        state = db.per_instance_state

        try:
            if not state.initialized:
                stop = False
                error_log = ""
                
                # Validate inputs
                if len(db.inputs.wheelRadius) == 0:
                    error_log += "Wheel radius list is empty\n"
                    stop = True
                if len(db.inputs.wheelPositions) == 0:
                    error_log += "Wheel positions list is empty\n"
                    stop = True
                if len(db.inputs.wheelRadius) != len(db.inputs.wheelPositions):
                    error_log += "Number of wheel radii must match number of wheel positions\n"
                    stop = True
                    
                if stop:
                    db.log_warning(error_log)
                    return False

                # Initialize controller parameters
                state.wheel_radius = db.inputs.wheelRadius
                state.wheel_positions = db.inputs.wheelPositions
                state.wheel_base_length = db.inputs.wheelBaseLength
                state.wheel_base_width = db.inputs.wheelBaseWidth
                state.up_axis = db.inputs.upAxis
                state.max_linear_speed = db.inputs.maxLinearSpeed
                state.max_angular_speed = db.inputs.maxAngularSpeed
                state.max_wheel_speed = db.inputs.maxWheelSpeed
                state.max_steering_speed = db.inputs.maxSteeringSpeed
                state.linear_gain = db.inputs.linearGain
                state.angular_gain = db.inputs.angularGain
                state.optimize_steering = db.inputs.optimizeSteeringAngles

                state.initialize_controller()

            # Compute wheel commands
            drive_velocities, steering_angles = state.forward(np.array(db.inputs.inputVelocity))

            # Set outputs
            db.outputs.driveVelocityCommand = drive_velocities.tolist()
            db.outputs.steeringAngleCommand = steering_angles.tolist()

        except Exception as error:
            db.log_warning(f"SwerveDriveController error: {str(error)}")
            return False

        return True

