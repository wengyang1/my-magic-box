import numpy as np

def rotate_point_cloud(batch_data, rot_angle):
    """
    Rotate 3D point cloud.
    Args:
        batch_data: nx3, float32. Original point clouds.
        rot_angle: float32. Rotation angle along up-axis.
    Returns:
        nx3, float32. Rotated point clouds.
    """
    # Rotation matrix
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rx = np.array([[1, 0, 0],
                   [0, cosval, -sinval],
                   [0, sinval, cosval]])
    return (np.dot(batch_data.reshape((-1, 3)), rx.T)).reshape(batch_data.shape)

def compute_box_3d(size, heading_angle, center):
    """
    Compute 3D bounding box corners.
    Args:
        size: tuple of (h, w, l), float32. 3D bounding box size.
        heading_angle: float32. Heading angle (yaw) in camera coordinate.
        center: tuple of (x, y, z), float32. 3D bounding box center.
    Returns:
        corners_3d: nx8x3, float32. 3D bounding box corners.
    """
    # 3D bounding box dimensions
    h, w, l = size
    # 3D bounding box center
    x, y, z = center
    # 8 corners of the 3D bounding box in its local coordinate system
    corners_3d = np.array([
        [l / 2, w / 2, -h / 2],
        [l / 2, -w / 2, -h / 2],
        [-l / 2, -w / 2, -h / 2],
        [-l / 2, w / 2, -h / 2],
        [l / 2, w / 2, h / 2],
        [l / 2, -w / 2, h / 2],
        [-l / 2, -w / 2, h / 2],
        [-l / 2, w / 2, h / 2]
    ])
    # Rotate the 3D bounding box
    corners_3d_rotated = rotate_point_cloud(corners_3d, heading_angle)
    # Translate the 3D bounding box to its center
    corners_3d_translated = corners_3d_rotated + np.array([x, y, z])
    return corners_3d_translated.reshape(1, 8, 3)

def parse_kitti_label(line):
    """
    Parse a single line from the KITTI label file.
    Args:
        line: str. A single line from the KITTI label file.
    Returns:
        type: str. Object type.
        truncated: float32. Truncation.
        occluded: int32. Occlusion.
        alpha: float32. Observation angle.
        bbox: tuple of (xmin, ymin, xmax, ymax), int32. 2D bounding box.
        dimensions: tuple of (height, width, length), float32. 3D object dimensions.
        location: tuple of (x, y, z), float32. 3D object location.
        rotation_y: float32. Rotation around the Y-axis in camera coordinates.
        score: float32. Detection score.
    """
    parts = line.strip().split()
    type = parts[0]
    truncated = float(parts[1])
    occluded = int(parts[2])
    alpha = float(parts[3])
    bbox = tuple(map(float, parts[4:8]))
    dimensions = tuple(map(float, parts[8:11]))
    location = tuple(map(float, parts[11:14]))
    rotation_y = float(parts[14])
    score = float(parts[15]) if len(parts) > 16 else -1
    return type, truncated, occluded, alpha, bbox, dimensions, location, rotation_y, score

def main():
    # Example usage with a hypothetical KITTI label file
    label_file = 'training/label_2/000008t.txt'  # Replace with your actual label file path
    with open(label_file, 'r') as f:
        for line in f:
            type, truncated, occluded, alpha, bbox, dimensions, location, rotation_y, score = parse_kitti_label(line)
            corners_3d = compute_box_3d(dimensions, rotation_y, location)
            print(corners_3d)

if __name__ == '__main__':
    main()