"""
Copyright (C) Microsoft Corporation. 
Copyright (C) 2025 IAMAI CONSULTING CORP
MIT License.

Demonstrates using a basic lidar sensor with a cylindrical scan pattern.
"""

import asyncio
import cv2

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
                    # Optional: display every 10th frame to reduce load
                    if len(rgb_frames) % 10 == 0:
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

        # Fly the drone around the scene
        projectairsim_log().info("Move up")
        move_task = await drone.move_by_velocity_async(
            v_north=0.0, v_east=0.0, v_down=-2.0, duration=5.0
        )
        await move_task

        projectairsim_log().info("Move north")
        move_task = await drone.move_by_velocity_async(
            v_north=4.0, v_east=0.0, v_down=0.0, duration=24.0
        )
        await move_task

        # projectairsim_log().info("Move north-west")
        # move_task = await drone.move_by_velocity_async(
        #     v_north=4.0, v_east=-4.0, v_down=0.0, duration=8.0
        # )
        # await move_task

        # projectairsim_log().info("Move north")
        # move_task = await drone.move_by_velocity_async(
        #     v_north=4.0, v_east=0.0, v_down=0.0, duration=3.0
        # )
        # await move_task

        projectairsim_log().info("Move down")
        move_task = await drone.move_by_velocity_async(
            v_north=0.0, v_east=0.0, v_down=2.0, duration=5.0
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
