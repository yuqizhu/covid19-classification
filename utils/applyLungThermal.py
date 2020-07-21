import cv2
import numpy as np


def applyLungThermal(thermal_iamge_path, mask_image_path, output_image_path,
                     pts_dict):
    """
    thermal_iamge_path is the path of the input thermal image
    Example: "input/demo_thermal.bmp"

    mask_image_path is the directory of the lung mask image
    Example: "input/mask.png"

    output_image_path is the path of the output lung area image
    Example: 'output/'

    pts_thermal is the key points in the order of left shoulder,
    throat, right shoulder
    Example: [[477, 332], [635, 275], [793, 318]]

    The output is a thermal image of the lung area
    """
    # Import thermal image and mask
    thermal = cv2.imread(thermal_iamge_path)
    mask = cv2.imread(mask_image_path)

    # Extract related thermal key point from key point dictionary
    nose = list(pts_dict[0])
    throat = list(pts_dict[1])
    left_shoulder = list(pts_dict[2])
    right_shoulder = list(pts_dict[5])

    throat[1] = int((nose[1] + throat[1]) / 2)
    left_shoulder[1] = int((left_shoulder[1] + right_shoulder[1]) / 2)
    right_shoulder[1] = int((left_shoulder[1] + right_shoulder[1]) / 2)

    # Measured mask key points based on CDCL dection output patterns
    pts_mask = np.float32([[200, 400], [1400, -200], [2600, 400]])
    pts_thermal = np.float32([left_shoulder, throat, right_shoulder])

    h_mask, w_mask, _ = mask.shape
    h_thermal, w_thermal, _ = thermal.shape

    # Use Affine Transformation to align mask and crop to thermal size
    M = cv2.getAffineTransform(pts_mask, pts_thermal)
    trans_mask = cv2.warpAffine(mask, M, (w_mask, h_mask))
    cropped_mask = trans_mask[0:h_thermal, 0:w_thermal]

    # Use mask to filter out lung area on the thermal image
    thermal_lung = cv2.bitwise_and(thermal, cropped_mask)

    # Check if Affine Transformation is successful
    if cv2.countNonZero(cv2.cvtColor(thermal_lung, cv2.COLOR_BGR2GRAY)) == 0:
        print("applyLungThermal Message: Lung Area Segmentation Failed")
    else:
        print("applyLungThermal Message: Lung Area Segmentation Succeeded")

    # Output the processed image
    out_name = thermal_iamge_path.split('/')[-1].split('.')[0] + '_lung.png'
    cv2.imwrite(output_image_path + out_name, thermal_lung)
