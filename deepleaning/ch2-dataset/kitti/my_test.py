import numpy as np
import cv2

def calculate_new_coordinates(original_width, original_height, target_width, target_height, corners_2d):
    # 计算裁剪区域的宽度和高度，以保持宽高比尽可能接近目标图像的宽高比
    target_aspect_ratio = target_width / target_height
    original_aspect_ratio = original_width / original_height

    if target_aspect_ratio > original_aspect_ratio:
        # 目标图像更宽，因此我们需要裁剪高度并缩放宽度
        crop_height = int(original_width / target_aspect_ratio)
        crop_width = original_width
        y_start = (original_height - crop_height) // 2
        x_start = 0
    else:
        # 目标图像更高或等比，因此我们需要裁剪宽度并缩放高度
        crop_width = int(original_height * target_aspect_ratio)
        crop_height = original_height
        x_start = (original_width - crop_width) // 2
        y_start = 0

    # 计算缩放比例
    scale_x = target_width / crop_width
    scale_y = target_height / crop_height

    # 计算新图像上的坐标
    corners_2d_new = np.zeros((len(corners_2d),2))
    corners_2d_new[:,0] =  (corners_2d[:,0] - x_start) * scale_x
    corners_2d_new[:,1] = (corners_2d[:,1] - y_start) * scale_y

    return corners_2d_new

def load_kitti_labels(label_file):
    """
    Load 3D bounding boxes from a KITTI label file (.txt).
    """
    boxes3d = []
    with open(label_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(' ')
            x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
            center = np.array([x, y, z])
            height, width, length = float(parts[8]), float(parts[9]), float(parts[10])
            yaw = float(parts[14])
            corners = compute_3d_box_corners(center, length, width, height, yaw)
            boxes3d.append(corners)
    return np.array(boxes3d)


def compute_3d_box_corners(center, length, width, height, yaw):
    """
    Compute the 3D bounding box corners from center, dimensions, and orientation.
    This is a simplified version that assumes no pitch and roll for demonstration.
    """
    # Create rotation matrix from yaw (around z-axis)
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    # Create the 8 corners of the 3D box in its local coordinate system
    x_corners = np.array(
        [-length / 2, length / 2, length / 2, -length / 2, -length / 2, length / 2, length / 2, -length / 2])
    y_corners = np.array([-width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2, width / 2])
    z_corners = np.array(
        [-height / 2, -height / 2, -height / 2, -height / 2, height / 2, height / 2, height / 2, height / 2])
    z_corners = z_corners + height/2
    # x_corners =x_corners+length/2
    # y_corners = y_corners + width/2
    corners_3d = np.vstack((x_corners, y_corners, z_corners)).T

    # Rotate the corners to the global coordinate system
    corners_3d_rotated = (Ry @ corners_3d.T).T

    # Translate the corners to the center location
    corners_3d_translated = corners_3d_rotated + center

    return corners_3d_translated


def load_kitti_calib(calib_file):
    """
    Load camera calibration parameters from a KITTI calib file (.txt).
    """
    calib_data = {}
    with open(calib_file, 'r') as f:
        for line in f.readlines():
            if line:
                key, value = line.split(':', 1)
                calib_data[key.strip()] = np.array([float(x) for x in value.strip().split()])
    return calib_data


def project_to_image(points_3d, P):
    """
    Project 3D points to 2D image plane using projection matrix P.
    """
    points_4d_homog = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))  # Add homogeneous coordinate
    points_2d_homog = (P @ points_4d_homog.T).T  # Project to 2D homogeneous coordinates
    points_2d = points_2d_homog[:, :2] / points_2d_homog[:, 2:3]  # Normalize to 2D image coordinates
    return points_2d


def main():
    label_file = 'training/label_2/000008t.txt'
    calib_file = 'training/calib/000008.txt'
    image_file = 'training/image_2/000008.png'  # Make sure this matches the timestamp of your label file

    # Load 3D bounding boxes from label file
    boxes3d = load_kitti_labels(label_file)

    # Load camera calibration parameters
    calib_data = load_kitti_calib(calib_file)
    P0 = np.reshape(calib_data['P0'], (3, 4))  # The P0 projection matrix from the calib file
    Tr_velo_to_cam = np.reshape(calib_data['Tr_velo_to_cam'], (3, 4))
    # Load the left grayscale image
    image = cv2.imread(image_file)

    # Project each 3D bounding box to the 2D image plane
    for box_3d in boxes3d:
        # Project the 8 corners of the 3D box
        corners_2d = project_to_image(box_3d, P0)

        # Clip to image boundaries
        corners_2d = np.clip(corners_2d, [0, 0], [image.shape[1] - 1, image.shape[0] - 1])
        # corners_2d=calculate_new_coordinates(1392,512,1224,370,corners_2d)
        # Draw the projected bounding box on the image (for visualization)
        corners_2d_int = corners_2d.astype(int)

        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            cv2.line(image, tuple(corners_2d_int[i]), tuple(corners_2d_int[j]), (255, 0, 0), 2)
            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(image, tuple(corners_2d_int[i]), tuple(corners_2d_int[j]), (255, 0, 0), 2)
            i, j = k, k + 4
            cv2.line(image, tuple(corners_2d_int[i]), tuple(corners_2d_int[j]), (255, 0, 0), 2)

    # Display or save the image with projected bounding boxes
    cv2.imshow('Projected 3D Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite('projected_boxes.png', image)  # Uncomment to save the image


if __name__ == "__main__":
    main()
