import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_comparison(joint_angles_3D_1, joint_angles_3D_2, 
                    angular_velocity_3D_1, angular_velocity_3D_2,
                    angular_acceleration_3D_1, angular_acceleration_3D_2,
                    linear_velocity_3D_1, linear_velocity_3D_2,
                    linear_acceleration_3D_1, linear_acceleration_3D_2):
    
    # Function to calculate norm for velocity and acceleration vectors
    def norm(vectors):
        return np.linalg.norm(vectors, axis=-1)

    # 1. Joint Angles (3, 2)
    fig, axs = plt.subplots(3, 2, figsize=(12, 14))
    fig.suptitle('3D Joint Angles Comparison', fontsize=16)

    # Left Leg (Left column)
    axs[0, 0].plot(joint_angles_3D_2["left_leg"]["ankle"], label="Before Kalman Filter - Ankle", color='blue')
    axs[0, 0].plot(joint_angles_3D_1["left_leg"]["ankle"], label="After Kalman Filter - Ankle", color='cyan')

    axs[0, 0].set_title("Left Leg - Ankle")
    axs[0, 0].legend()
    axs[0, 0].set_ylabel("Angle (rad)")
    axs[0, 0].set_xlabel("Frame")


    axs[1, 0].plot(joint_angles_3D_2["left_leg"]["knee"], label="Before Kalman Filter - Knee", color='green')
    axs[1, 0].plot(joint_angles_3D_1["left_leg"]["knee"], label="After Kalman Filter - Knee", color='lime')

    axs[1, 0].set_title("Left Leg - Knee")
    axs[1, 0].legend()
    axs[1, 0].set_ylabel("Angle (rad)")
    axs[1, 0].set_xlabel("Frame")

    axs[2, 0].plot(joint_angles_3D_2["left_leg"]["hip"], label="Before Kalman Filter - Hip", color='red')
    axs[2, 0].plot(joint_angles_3D_1["left_leg"]["hip"], label="After Kalman Filter - Hip", color='orange')

    axs[2, 0].set_title("Left Leg - Hip")
    axs[2, 0].legend()
    axs[2, 0].set_ylabel("Angle (rad)")
    axs[2, 0].set_xlabel("Frame")

    # Right Leg (Right column)
    axs[0, 1].plot(joint_angles_3D_2["right_leg"]["ankle"], label="Before Kalman Filter - Ankle", color='blue')
    axs[0, 1].plot(joint_angles_3D_1["right_leg"]["ankle"], label="After Kalman Filter - Ankle", color='cyan')

    axs[0, 1].set_title("Right Leg - Ankle")
    axs[0, 1].legend()
    axs[0, 1].set_ylabel("Angle (rad)")
    axs[0, 1].set_xlabel("Frame")

    axs[1, 1].plot(joint_angles_3D_2["right_leg"]["knee"], label="Before Kalman Filter - Knee", color='green')
    axs[1, 1].plot(joint_angles_3D_1["right_leg"]["knee"], label="After Kalman Filter - Knee", color='lime')

    axs[1, 1].set_title("Right Leg - Knee")
    axs[1, 1].legend()
    axs[1, 1].set_ylabel("Angle (rad)")
    axs[1, 1].set_xlabel("Frame")

    axs[2, 1].plot(joint_angles_3D_2["right_leg"]["hip"], label="Before Kalman Filter - Hip", color='red')
    axs[2, 1].plot(joint_angles_3D_1["right_leg"]["hip"], label="After Kalman Filter - Hip", color='orange')

    axs[2, 1].set_title("Right Leg - Hip")
    axs[2, 1].legend()
    axs[2, 1].set_ylabel("Angle (rad)")
    axs[2, 1].set_xlabel("Frame")

    plt.tight_layout()
    plt.show()

    # 2. Angular Velocities (3, 2)
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle('Angular Velocities', fontsize=16)

    # Left Leg (Left column)
    axs[0, 0].plot(angular_velocity_3D_1["left_leg"]["ankle"], label="After Kalman Filter - Ankle", color='cyan')
    axs[0, 0].plot(angular_velocity_3D_2["left_leg"]["ankle"], label="Before Kalman Filter - Ankle", color='blue')
    axs[0, 0].set_title("Left Leg - Ankle")
    axs[0, 0].legend()


    axs[1, 0].plot(angular_velocity_3D_1["left_leg"]["knee"], label="After Kalman Filter - Knee", color='lime')
    axs[1, 0].plot(angular_velocity_3D_2["left_leg"]["knee"], label="Before Kalman Filter - Knee", color='green')
    axs[1, 0].set_title("Left Leg - Knee")
    axs[1, 0].legend()

    axs[2, 0].plot(angular_velocity_3D_1["left_leg"]["hip"], label="After Kalman Filter - Hip", color='orange')
    axs[2, 0].plot(angular_velocity_3D_2["left_leg"]["hip"], label="Before Kalman Filter - Hip", color='red')
    axs[2, 0].set_title("Left Leg - Hip")
    axs[2, 0].legend()

    # Right Leg (Right column)
    axs[0, 1].plot(angular_velocity_3D_1["right_leg"]["ankle"], label="After Kalman Filter - Ankle", color='cyan')
    axs[0, 1].plot(angular_velocity_3D_2["right_leg"]["ankle"], label="Before Kalman Filter - Ankle", color='blue')
    axs[0, 1].set_title("Right Leg - Ankle")
    axs[0, 1].legend()

    axs[1, 1].plot(angular_velocity_3D_1["right_leg"]["knee"], label="After Kalman Filter - Knee", color='lime')
    axs[1, 1].plot(angular_velocity_3D_2["right_leg"]["knee"], label="Before Kalman Filter - Knee", color='green')
    axs[1, 1].set_title("Right Leg - Knee")
    axs[1, 1].legend()

    axs[2, 1].plot(angular_velocity_3D_1["right_leg"]["hip"], label="After Kalman Filter - Hip", color='orange')
    axs[2, 1].plot(angular_velocity_3D_2["right_leg"]["hip"], label="Before Kalman Filter - Hip", color='red')
    axs[2, 1].set_title("Right Leg - Hip")
    axs[2, 1].legend()

    plt.tight_layout()
    plt.show()

    # 3. Angular Accelerations (3, 2)
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle('Angular Accelerations', fontsize=16)

    # Left Leg (Left column)
    axs[0, 0].plot(angular_acceleration_3D_1["left_leg"]["ankle"], label="After Kalman Filter - Ankle", color='cyan')
    axs[0, 0].plot(angular_acceleration_3D_2["left_leg"]["ankle"], label="Before Kalman Filter - Ankle", color='blue')
    axs[0, 0].set_title("Left Leg - Ankle")
    axs[0, 0].legend()

    axs[1, 0].plot(angular_acceleration_3D_1["left_leg"]["knee"], label="After Kalman Filter - Knee", color='lime')
    axs[1, 0].plot(angular_acceleration_3D_2["left_leg"]["knee"], label="Before Kalman Filter - Knee", color='green')
    axs[1, 0].set_title("Left Leg - Knee")
    axs[1, 0].legend()

    axs[2, 0].plot(angular_acceleration_3D_1["left_leg"]["hip"], label="After Kalman Filter - Hip", color='orange')
    axs[2, 0].plot(angular_acceleration_3D_2["left_leg"]["hip"], label="Before Kalman Filter - Hip", color='red')
    axs[2, 0].set_title("Left Leg - Hip")
    axs[2, 0].legend()

    # Right Leg (Right column)
    axs[0, 1].plot(angular_acceleration_3D_1["right_leg"]["ankle"], label="After Kalman Filter - Ankle", color='cyan')
    axs[0, 1].plot(angular_acceleration_3D_2["right_leg"]["ankle"], label="Before Kalman Filter - Ankle", color='blue')
    axs[0, 1].set_title("Right Leg - Ankle")
    axs[0, 1].legend()

    axs[1, 1].plot(angular_acceleration_3D_1["right_leg"]["knee"], label="After Kalman Filter - Knee", color='lime')
    axs[1, 1].plot(angular_acceleration_3D_2["right_leg"]["knee"], label="Before Kalman Filter - Knee", color='green')
    axs[1, 1].set_title("Right Leg - Knee")
    axs[1, 1].legend()

    axs[2, 1].plot(angular_acceleration_3D_1["right_leg"]["hip"], label="After Kalman Filter - Hip", color='orange')
    axs[2, 1].plot(angular_acceleration_3D_2["right_leg"]["hip"], label="Before Kalman Filter - Hip", color='red')
    axs[2, 1].set_title("Right Leg - Hip")
    axs[2, 1].legend()

    plt.tight_layout()
    plt.show()

    # 4. Linear Velocity (2, 2)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Linear Velocities', fontsize=16)

    # Left Leg (Left column)
    axs[0, 0].plot(linear_velocity_3D_1["left_leg"]["ankle"], label="After Kalman Filter - Ankle", color='cyan')
    axs[0, 0].plot(linear_velocity_3D_2["left_leg"]["ankle"], label="Before Kalman Filter - Ankle", color='blue')
    axs[0, 0].set_title("Left Leg - Ankle")
    axs[0, 0].legend()

    axs[1, 0].plot(linear_velocity_3D_1["left_leg"]["knee"], label="After Kalman Filter - Knee", color='lime')
    axs[1, 0].plot(linear_velocity_3D_2["left_leg"]["knee"], label="Before Kalman Filter - Knee", color='green')
    axs[1, 0].set_title("Left Leg - Knee")
    axs[1, 0].legend()

    # Right Leg (Right column)
    axs[0, 1].plot(linear_velocity_3D_1["right_leg"]["ankle"], label="After Kalman Filter - Ankle", color='cyan')
    axs[0, 1].plot(linear_velocity_3D_2["right_leg"]["ankle"], label="Before Kalman Filter - Ankle", color='blue')
    axs[0, 1].set_title("Right Leg - Ankle")
    axs[0, 1].legend()

    axs[1, 1].plot(linear_velocity_3D_1["right_leg"]["knee"], label="After Kalman Filter - Knee", color='lime')
    axs[1, 1].plot(linear_velocity_3D_2["right_leg"]["knee"], label="Before Kalman Filter - Knee", color='green')
    axs[1, 1].set_title("Right Leg - Knee")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

    # 5. Linear Acceleration (2, 2) (same structure as Linear Velocity)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Linear Accelerations', fontsize=16)

    # Left Leg (Left column)
    axs[0, 0].plot(linear_acceleration_3D_1["left_leg"]["ankle"], label="After Kalman Filter - Ankle", color='cyan')
    axs[0, 0].plot(linear_acceleration_3D_2["left_leg"]["ankle"], label="Before Kalman Filter - Ankle", color='blue')
    axs[0, 0].set_title("Left Leg - Ankle")
    axs[0, 0].legend()

    axs[1, 0].plot(linear_acceleration_3D_1["left_leg"]["knee"], label="After Kalman Filter - Knee", color='lime')
    axs[1, 0].plot(linear_acceleration_3D_2["left_leg"]["knee"], label="Before Kalman Filter - Knee", color='green')
    axs[1, 0].set_title("Left Leg - Knee")
    axs[1, 0].legend()

    # Right Leg (Right column)
    axs[0, 1].plot(linear_acceleration_3D_1["right_leg"]["ankle"], label="After Kalman Filter - Ankle", color='cyan')
    axs[0, 1].plot(linear_acceleration_3D_2["right_leg"]["ankle"], label="Before Kalman Filter - Ankle", color='blue')
    axs[0, 1].set_title("Right Leg - Ankle")
    axs[0, 1].legend()

    axs[1, 1].plot(linear_acceleration_3D_1["right_leg"]["knee"], label="After Kalman Filter - Knee", color='lime')
    axs[1, 1].plot(linear_acceleration_3D_2["right_leg"]["knee"], label="Before Kalman Filter - Knee", color='green')
    axs[1, 1].set_title("Right Leg - Knee")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()
