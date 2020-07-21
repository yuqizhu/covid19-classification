import cv2
import imutils
import numpy as np
from imutils import contours
from skimage import measure


def circleInflamArea(input_image_path, input_temp_path, output_image_path,
                     inflam_thresh_temp, grey_threshold=50, area_threshold=500):
    """
    input_image_path is the path of the segment lung thermal image
    Example: "output/demo_thermal_lung.jpg"

    input_temp_path is the path of text file that contains pixel-wise
    temperature information
    Example: "input/demo_temp.txt"

    output_image_path is the path of the lung thermal image with
    inflammation area circled
    Example: "output/"

    inflam_thresh_temp is the temperature value represents the minimum
    temperature that is considered as inflammation
    Example: 34.2

    grey_threshold is the threshold of grey channel value used to filter
    potential noise of the inflammation area selection
    Exmaple: 50

    area_threshold is the threshold of pixel number required for an area
    to be considered as inflammation area
    Example: 300
    """
    # Import thermal image and pixel-wise temperature information file
    inflam_lung = cv2.imread(input_image_path)
    temp = np.loadtxt(input_temp_path)

    # Filter pixels that have higher than threshold temperature
    mask = temp > inflam_thresh_temp
    gray = cv2.cvtColor(inflam_lung, cv2.COLOR_BGR2GRAY)
    cropped = gray * mask

    # Process image to create a mask of inflammation area
    blurred = cv2.GaussianBlur(cropped, (11, 11), 0)
    thresh = cv2.threshold(blurred, grey_threshold, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)

    # Process image to reduce noise of the mask
    labels = measure.label(thresh, connectivity=2, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")
    for label in np.unique(labels):
        if label == 0:
            continue

        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        if numPixels > area_threshold:
            mask = cv2.add(mask, labelMask)

    # Find the contour of each inflammation area and draw circles if found any contours
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) > 0:
        print("circleInflamArea Message: Found %s Inflammable Area(s)" % len(cnts))
        cnts = contours.sort_contours(cnts)[0]

        for c in cnts:
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            cv2.circle(inflam_lung, (int(cX), int(cY)), int(radius) + 10, (0, 0, 255), 2)
    else:
        print("circleInflamArea Message: No Inflammation Area Found")

    # Output the processed image
    out_name = input_image_path.split('/')[-1].split('.')[0] + '_circled.png'
    cv2.imwrite(output_image_path + out_name, inflam_lung)
