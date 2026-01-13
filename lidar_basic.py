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

from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.utils import projectairsim_log, unpack_image
from projectairsim.image_utils import ImageDisplay
from projectairsim.types import ImageType
# from projectairsim.lidar_utils import LidarDisplay


# Async main function to wrap async drone commands
async def main():
    # Create a Project AirSim client
    client = ProjectAirSimClient()

    # Initialize an ImageDisplay object to display camera sub-windows
    image_display = ImageDisplay(
        num_subwin=3,
        screen_res_x=2880,
        screen_res_y=1800,
        subwin_y_pct=0.1)
    # List to store frames for video
    rgb_frames = []
    segmentation_frames = []
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

        # ------------------------------------------------------------------------------

        # client.subscribe(
        #     drone.sensors["sf45b_lidar"]["lidar"],
        #     lambda _, lidar: lidar_display.receive(lidar),
        # )

        # lidar_display.start()

        # ------------------------------------------------------------------------------

        # Set the drone to be ready to fly
        drone.enable_api_control()
        drone.arm()

        # Start image capture task
        capture_task = asyncio.create_task(capture_images())

        # Take off
        projectairsim_log().info("Taking off")
        move_task = await drone.move_by_velocity_async(
            v_north=0.0, v_east=0.0, v_down=-2.0, duration=5.0, yaw=0.0, yaw_is_rate=True
        )
        await move_task

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

    except Exception as err:
        projectairsim_log().error(f"Exception occurred: {err}", exc_info=True)

    finally:
        # Always disconnect from the simulation environment to allow next connection
        client.disconnect()

        image_display.stop()

        # ------------------------------------------------------------------------------

        # lidar_display.stop()

        # ------------------------------------------------------------------------------


if __name__ == "__main__":
    asyncio.run(main())  # Runner for async main function
