import cv2
import numpy as np

# Load the main image
# This is the image where the red regions will be detected and replaced.
image = cv2.imread("project\\Resources\\lapi.jpg")

# Load the replacement image
# This is the image that will replace the red regions in the main image.
replacement_image = cv2.imread("Open CV\\resources\\image.jpg")

# Convert the main image to HSV color space
# HSV color space is better suited for color detection than RGB.
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the red color range in HSV
# Red color in HSV has two distinct ranges due to its circular nature in the hue space.
lower_red1 = np.array([0, 120, 70])  # Lower bound for red
upper_red1 = np.array([10, 255, 255])  # Upper bound for red
lower_red2 = np.array([170, 120, 70])  # Lower bound for red
upper_red2 = np.array([180, 255, 255])  # Upper bound for red

# Create masks for red color
# Combine two masks to detect all shades of red in the defined ranges.
mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)

# Find contours for the red regions
# Contours help in identifying the boundaries of red areas in the image.
contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define a minimum area threshold to ignore small, insignificant objects.
min_area_threshold = 3000  # Area in pixels; adjust based on your image.

# Loop through detected contours
for contour in contours:
    # Filter out minor objects based on the area
    area = cv2.contourArea(contour)
    if area < min_area_threshold:
        continue  # Skip contours that are too small

    # Create a blank mask for the detected shape
    shape_mask = np.zeros_like(image, dtype=np.uint8)

    # Draw the detected contour as a filled shape on the mask
    cv2.drawContours(shape_mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    # Isolate the detected region in the original image
    object_mask = cv2.inRange(shape_mask, (255, 255, 255), (255, 255, 255))

    # Get the bounding box of the contour
    # This will be used to resize the replacement image to fit the detected region.
    x, y, w, h = cv2.boundingRect(contour)
    resized_replacement = cv2.resize(replacement_image, (w, h))

    # Create a mask for the resized replacement image
    resized_mask = object_mask[y:y+h, x:x+w]

    # Overlay the replacement image on the original image
    # 1. Extract the part of the resized replacement image corresponding to the mask
    replacement_cropped = cv2.bitwise_and(resized_replacement, resized_replacement, mask=resized_mask)

    # 2. Remove the detected region from the original image
    background = cv2.bitwise_and(image[y:y+h, x:x+w], image[y:y+h, x:x+w], mask=cv2.bitwise_not(resized_mask))

    # 3. Combine the background and the replacement image
    result = cv2.add(background, replacement_cropped)

    # 4. Place the result back into the original image
    image[y:y+h, x:x+w] = result

# Display the final image
cv2.imshow('Shape-Fitted Replacement (No Black Frame)', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
