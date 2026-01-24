import numpy as np

def pixel_to_ground_distance(h, alpha_deg, vfov_deg, hfov_deg, R_y, R_x, i, j):
    """
    Calculate ground coordinates for a pixel in the camera image.

    Parameters:
    -----------
    h : float
        Altitude (meters)
    alpha_deg : float
        Camera tilt angle (degrees) forward from Nadir (0° = straight down)
    vfov_deg : float
        Vertical Field of View (degrees)
    hfov_deg : float
        Horizontal Field of View (degrees)
    R_y : int
        Vertical resolution (total pixels in height)
    R_x : int
        Horizontal resolution (total pixels in width)
    i : int or array
        Horizontal pixel coordinate (0 at left, R_x at right)
    j : int or array
        Vertical pixel coordinate (0 at top, R_y at bottom)

    Returns:
    --------
    x_ground : float or array
        Lateral distance (left/right) from drone centerline (meters)
        Positive = right, Negative = left
    y_ground : float or array
        Longitudinal distance (forward/backward) from point beneath drone (meters)
        Positive = forward, Negative = backward
    D_ground : float or array
        Total Euclidean distance on ground from point beneath drone (meters)
    """

    # Convert angles to radians
    alpha = np.radians(alpha_deg)
    vfov = np.radians(vfov_deg)
    hfov = np.radians(hfov_deg)

    # Step 1: Calculate pixel angles relative to camera center
    phi_y = ((R_y / 2.0 - j) / R_y) * vfov
    phi_x = ((i - R_x / 2.0) / R_x) * hfov

    # Step 2: Calculate total pitch angle to ground point
    theta = alpha + phi_y

    # Step 3: Calculate ground coordinates
    # Longitudinal distance (forward/backward)
    y_ground = h * np.tan(theta)

    # Lateral distance (left/right)
    x_ground = (h / np.cos(theta)) * np.tan(phi_x)

    # Step 4: Total ground distance
    D_ground = np.sqrt(x_ground**2 + y_ground**2)

    return x_ground, y_ground, D_ground


# Example usage with your camera specs
# FOV: 128.2°(D), 104.6°(H), 61.6°(V)
h = 10.0  # 10 meters altitude
alpha_deg = 15.0  # 15° forward tilt
vfov_deg = 61.6  # Vertical FOV
hfov_deg = 104.6  # Horizontal FOV
R_y = 1080  # Example resolution (height)
R_x = 1920  # Example resolution (width)

print("=" * 70)
print("Ground Distance Calculator for Drone Camera")
print("=" * 70)
print(f"\nCamera Configuration:")
print(f"  Altitude: {h} m")
print(f"  Camera Tilt: {alpha_deg}° forward from Nadir")
print(f"  Vertical FOV: {vfov_deg}°")
print(f"  Horizontal FOV: {hfov_deg}°")
print(f"  Resolution: {R_x} × {R_y} pixels")
print("\n" + "=" * 70)

# Test key points in the image
test_points = [
    ("Top-Center", R_x // 2, 0),
    ("Center", R_x // 2, R_y // 2),
    ("Bottom-Center", R_x // 2, R_y - 1),
    ("Top-Left", 0, 0),
    ("Top-Right", R_x - 1, 0),
    ("Bottom-Left", 0, R_y - 1),
    ("Bottom-Right", R_x - 1, R_y - 1),
]

print("\nGround Distances for Key Image Points:")
print("-" * 70)
print(f"{'Point':<15} {'Pixel (i,j)':<15} {'X (m)':<10} {'Y (m)':<10} {'Distance (m)':<12}")
print("-" * 70)

for name, i, j in test_points:
    x_g, y_g, d_g = pixel_to_ground_distance(h, alpha_deg, vfov_deg, hfov_deg, R_y, R_x, i, j)
    print(f"{name:<15} ({i:4d},{j:4d})     {x_g:>7.2f}    {y_g:>7.2f}    {d_g:>7.2f}")

print("=" * 70)

# Example: Calculate for a blob detected by DNN
print("\n" + "=" * 70)
print("Example: ROI Blob Detection")
print("=" * 70)
blob_center_i = 960  # Center horizontally
blob_center_j = 200  # Near top of image (ROI ahead)

x_blob, y_blob, d_blob = pixel_to_ground_distance(
    h, alpha_deg, vfov_deg, hfov_deg, R_y, R_x, blob_center_i, blob_center_j
)

print(f"\nBlob detected at pixel ({blob_center_i}, {blob_center_j})")
print(f"  → Lateral offset (X): {x_blob:.2f} m {'(right)' if x_blob > 0 else '(left)' if x_blob < 0 else '(centered)'}")
print(f"  → Forward distance (Y): {y_blob:.2f} m")
print(f"  → Total ground distance: {d_blob:.2f} m")
print("\n" + "=" * 70)