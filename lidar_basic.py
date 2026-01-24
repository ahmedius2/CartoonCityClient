"""
Copyright (C) Microsoft Corporation. 
Copyright (C) 2025 IAMAI CONSULTING CORP
MIT License.

Demonstrates using a basic lidar sensor with a cylindrical scan pattern.
"""

import asyncio
import cv2
import threading
from pynput import keyboard
import time
import json
import math

from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.utils import projectairsim_log, unpack_image
from projectairsim.image_utils import ImageDisplay
from projectairsim.types import ImageType

# Configuration: Set to True to follow waypoints, False for manual control
USE_WAYPOINTS = True

waypoints_file_path = "D:/UnrealProjects/CartoonCityAirsim/Saved/waypoints_world_ue.json"

def load_waypoints(file_path):
    """Load waypoints from JSON file."""
    try:
        with open(file_path, 'r') as f:
            waypoints = json.load(f)
        projectairsim_log().info(f"Loaded {len(waypoints)} waypoints from {file_path}")
        return waypoints
    except Exception as e:
        projectairsim_log().error(f"Error loading waypoints: {e}")
        return []

def convert_ue_to_airsim(ue_x, ue_y, ue_z):
    """
    Convert UE coordinates (cm) to AirSim NED coordinates (m).
    UE: X=forward, Y=right, Z=up (in cm)
    AirSim NED: X=north, Y=east, Z=down (in m, positive down)
    """
    # Convert cm to meters and flip coordinate system
    # UE X (forward) -> AirSim X (north)
    # UE Y (right) -> AirSim Y (east)
    # UE Z (up, positive) -> AirSim Z (down, negative)
    x_ned = ue_x / 100.0  # North (forward)
    y_ned = ue_y / 100.0  # East (right)
    z_ned = -ue_z / 100.0  # Down (negative of up)
    return x_ned, y_ned, z_ned

def calculate_heading_and_distance(current_pos, target_pos):
    """
    Calculate heading (radians) and distance (meters) from current to target position.
    """
    dx = target_pos['x'] - current_pos['x']  # North component
    dy = target_pos['y'] - current_pos['y']  # East component
    dz = target_pos['z'] - current_pos['z']  # Down component

    # Horizontal distance
    distance_2d = math.sqrt(dx**2 + dy**2)
    # 3D distance
    distance_3d = math.sqrt(dx**2 + dy**2 + dz**2)

    # Calculate heading in degrees (0 = North, 90 = East)
    heading_rad = math.atan2(dy, dx)

    return heading_rad, distance_2d, distance_3d

async def follow_waypoints(drone, waypoints, waypoint_speed=3.0, position_tolerance=1.0):
    """
    Navigate drone through waypoints sequentially.
    waypoint_speed: Speed in m/s
    position_tolerance: Distance threshold to consider waypoint reached (meters)
    """
    if not waypoints:
        projectairsim_log().warning("No waypoints to follow")
        return

    projectairsim_log().info(f"Starting waypoint following: {len(waypoints)} waypoints")

    for i, waypoint in enumerate(waypoints):
        wp_label = waypoint.get('label', f'waypoint_{i+1}')
        wp_loc = waypoint['location_ue_cm']

        # Convert UE coordinates to AirSim NED coordinates
        target_x, target_y, target_z = convert_ue_to_airsim(
            wp_loc['x'], wp_loc['y'], wp_loc['z']
        )

        projectairsim_log().info(
            f"Navigating to {wp_label} ({i+1}/{len(waypoints)}): "
            f"Target position NED: ({target_x:.2f}, {target_y:.2f}, {target_z:.2f}) m"
        )

        # Navigate to waypoint
        reached = False
        max_attempts = 100  # Prevent infinite loops
        attempt = 0

        while not reached and attempt < max_attempts:
            # Get current position
            pose = drone.get_ground_truth_pose()
            current_pos = {
                'x': pose['translation']['x'],
                'y': pose['translation']['y'],
                'z': pose['translation']['z']
            }

            # Calculate heading and distance to target
            target_pos = {'x': target_x, 'y': target_y, 'z': target_z}
            heading, distance_2d, distance_3d = calculate_heading_and_distance(
                current_pos, target_pos
            )

            # Check if we're close enough
            if distance_3d < position_tolerance:
                projectairsim_log().info(f"Reached {wp_label} (distance: {distance_3d:.2f} m)")
                reached = True
                break

            # Calculate vertical velocity component
            dz = target_z - current_pos['z']
            v_down = 0.0
            if abs(dz) > 0.5:  # If vertical difference is significant
                v_down = dz * 0.5
                v_down = max(-1.0, min(1.0, v_down))  # Limit vertical velocity
            # Move towards waypoint
            duration = 1 #min(2.0, distance_2d / waypoint_speed)  # Adjust duration based on distance
            await drone.move_by_heading_async(
                heading=heading,
                speed=waypoint_speed,
                v_down=v_down,
                duration=duration,
                yaw_rate=0.3  # Yaw rotation rate in degrees per second
            )

            await asyncio.sleep(0.1)  # Small delay for position update
            attempt += 1

        if not reached:
            projectairsim_log().warning(f"Could not reach {wp_label} after {max_attempts} attempts")

        # Brief pause at waypoint
        await asyncio.sleep(0.5)

    projectairsim_log().info("Completed waypoint following")

# Async main function to wrap async drone commands
async def main():
    # Create a Project AirSim client
    client = ProjectAirSimClient()

    # Initialize an ImageDisplay object to display camera sub-windows
    image_display = ImageDisplay(
        num_subwin=3,
        screen_res_x=1920,
        screen_res_y=1080,
        subwin_y_pct=0.1)
    # List to store frames for video
    rgb_frames = []
    segmentation_frames = []
    pose_list = []
    capture_task = None
    stop_capture = False

    async def capture_images():
        while not stop_capture:
            try:
                images = drone.get_images("DownCamera", \
                                          [ImageType.SCENE, ImageType.SEGMENTATION])
                if len(images) >= 2 and len(images[ImageType.SCENE]) > 0 and \
                            len(images[ImageType.SEGMENTATION]) > 0:
                    rgb_frames.append(images[ImageType.SCENE])
                    segmentation_frames.append(images[ImageType.SEGMENTATION])
                    pose = drone.get_ground_truth_pose()
                    trl, rot = pose['translation'], pose['rotation']
                    pose = [trl['x'], trl['y'], trl['z'], rot['w'], rot['x'], rot['y'], rot['z']]
                    pose_list.append([float(x) for x in pose])
                    # Optional: display every 5th frame to reduce load
                    if len(rgb_frames) % 5 == 0:
                        # IDK if I need to unpack again here
                        image_display.receive(rgb_frames[-1], rgb_name)
                        image_display.receive(segmentation_frames[-1], seg_name)
            except Exception as e:
                projectairsim_log().error(f"Error capturing images: {e}")
            await asyncio.sleep(0.04)  # Match capture interval

    try:
        # Connect to simulation environment
        client.connect()

        # Create a World object to interact with the sim world and load a scene
        world = World(client, "scene_lidar_drone.jsonc", delay_after_load_sec=2)

        # Create a Drone object to interact with a drone in the loaded sim world
        drone = Drone(client, world, "Drone1")

        # Initialize display
        rgb_name = "RGB-Image"
        image_display.add_image(rgb_name, subwin_idx=0)
        seg_name = "Segmented-Image"
        image_display.add_image(seg_name, subwin_idx=1)
        image_display.start()

        # Set the drone to be ready to fly
        drone.enable_api_control()
        drone.arm()

        # Start image capture task
        capture_task = asyncio.create_task(capture_images())

        # Load waypoints if using waypoint mode
        waypoints = []
        takeoff_altitude_m = 10.0  # Default altitude if no waypoints

        if USE_WAYPOINTS:
            waypoints = load_waypoints(waypoints_file_path)
            if waypoints:
                # Get altitude from first waypoint (convert from cm to m, and UE Z to AirSim Z)
                first_wp_z_cm = waypoints[0]['location_ue_cm']['z']
                takeoff_altitude_m = -first_wp_z_cm / 100.0  # Negative because AirSim Z is down
                projectairsim_log().info(f"Takeoff altitude set to {takeoff_altitude_m:.2f} m from first waypoint")
            else:
                projectairsim_log().warning("No waypoints loaded, using default altitude")

        # Take off to waypoint altitude
        projectairsim_log().info(f"Taking off to altitude: {takeoff_altitude_m:.2f} m")

        # Get current position
        pose = drone.get_ground_truth_pose()
        current_z = pose['translation']['z']
        target_z = takeoff_altitude_m

        # Calculate vertical distance to travel
        dz = target_z - current_z

        projectairsim_log().info(f"Current Z: {current_z:.2f} m, Target Z: {target_z:.2f} m, dz: {dz:.2f} m")

        if abs(dz) > 0.1:  # Only move if significant difference
            # Calculate duration based on distance (assuming 2 m/s vertical speed)
            duration = abs(dz) / 2.0
            # In AirSim NED: v_down positive = down, v_down negative = up
            # dz negative means target is above (more negative Z), so we need negative v_down (up)
            # dz positive means target is below (less negative Z), so we need positive v_down (down)
            v_down = dz / duration if duration > 0 else 0.0
            v_down = max(-2.0, min(2.0, v_down))  # Limit vertical velocity

            projectairsim_log().info(f"Takeoff: v_down={v_down:.2f} m/s, duration={duration:.2f} s")

            move_task = await drone.move_by_velocity_async(
                v_north=0.0, v_east=0.0, v_down=v_down, duration=duration, yaw=0.0, yaw_is_rate=True
            )
            await move_task
        else:
            projectairsim_log().info("Already at target altitude, skipping takeoff")

        projectairsim_log().info("Takeoff complete")

        if USE_WAYPOINTS and waypoints:
            # Follow waypoints
            await follow_waypoints(drone, waypoints, waypoint_speed=3.0, position_tolerance=2.0)
        else:
            # Manual control variables
            velocity_forward = 2.0
            yaw = 0.0
            land = False

            def on_press(key):
                nonlocal velocity_forward, yaw, land
                try:
                    if key == keyboard.Key.up:
                        velocity_forward += 0.5
                    elif key == keyboard.Key.down:
                        velocity_forward = max(0, velocity_forward - 0.5)
                    elif key == keyboard.Key.left:
                        yaw += -0.3
                    elif key == keyboard.Key.right:
                        yaw += 0.3
                    elif hasattr(key, 'char') and key.char.lower() == 'l':
                        land = True
                except AttributeError:
                    pass

            def on_release(key):
                pass
                #nonlocal yaw
                #if key in (keyboard.Key.left, keyboard.Key.right):
                #    yaw = 0.0

            listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            listener.start()

            # Manual control loop
            while not land:
                # print(f"Velocity Forward: {velocity_forward}, Yaw Rate: {yaw}, Heading: {yaw}")
                # t1 = time.time()
                await drone.move_by_heading_async(
                    heading=yaw, speed=velocity_forward, v_down=0.0, duration=1.0, yaw_rate=0.3
                )
                await asyncio.sleep(0.5)
                # t2 = time.time()
                # projectairsim_log().info(f"Control loop iteration took {t2 - t1:.2f} seconds")

            listener.stop()

        # Land
        projectairsim_log().info("Landing")
        move_task = await drone.move_by_velocity_async(
            v_north=0.0, v_east=0.0, v_down=4.0, duration=5.0, yaw=0.0, yaw_is_rate=True
        )
        await move_task

        # Shut down the drone
        drone.disarm()
        drone.disable_api_control()

        # Stop capture
        stop_capture = True
        if capture_task:
            await capture_task

        # Write frames to video files
        for frames, fname in ((rgb_frames, 'rgb_input.avi'), (segmentation_frames, 'segmentation_ground_truth.avi')):
            if not frames:
                projectairsim_log().warning(f"No frames to write for {fname}")
                continue
            frames = [unpack_image(img) for img in frames]
            height, width = frames[0].shape[:2]
            writer = cv2.VideoWriter(
                fname,
                cv2.VideoWriter_fourcc(*'MJPG'),
                25,  # fps, based on capture-interval 0.04
                (width, height)
            )
            for frame in frames:
                writer.write(frame)
            writer.release()
            projectairsim_log().info(f"Saved {len(frames)} frames to {fname}")

        with open("poses.json", "w") as f:
            json.dump(pose_list, f, indent=2)

    except Exception as err:
        projectairsim_log().error(f"Exception occurred: {err}", exc_info=True)

    finally:
        # Always disconnect from the simulation environment to allow next connection
        client.disconnect()

        image_display.stop()

if __name__ == "__main__":
    asyncio.run(main())  # Runner for async main function