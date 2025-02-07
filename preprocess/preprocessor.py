import cv2
import numpy as np

CLAHE_CLIP_LIMIT = 2.0  # Clip limit for CLAHE contrast enhancement.
CLAHE_TITLE_GRID_SIZE = (5, 5)  # Tile grid size for CLAHE.
ROI_SIZE = (276, 276)  # Size of the Region of Interest (ROI) to extract (width, height).
TARGET_SIZE = (138, 138)  # Target size to resize the ROI to (width, height).
NOISE_REDUCTION_KERNEL_SIZE = 1  # Kernel size for Gaussian blur (if noise reduction is used).
THRESHOLD_VALUE = 80  # Threshold value for palm segmentation.

contrast = cv2.createCLAHE(CLAHE_CLIP_LIMIT, CLAHE_TITLE_GRID_SIZE)


def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def enhance_contrast_clahe(gray_image):
    return contrast.apply(gray_image)


def reduce_noise(image):
    return cv2.medianBlur(image, NOISE_REDUCTION_KERNEL_SIZE, 0)


def segment_palm(enhanced_gray_image):
    _, binary_image = cv2.threshold(enhanced_gray_image, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return binary_image, contours


def extract_roi_centroid(gray_image, contours):
    """Extracts the Region of Interest (ROI) based on the largest contour's centroid."""
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:  # to avoid division by zero
        return None  # contour area is zero, handle appropriately

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    roi_width, roi_height = ROI_SIZE
    x = max(0, cx - roi_width // 2)  # Ensure ROI is within image bounds
    y = max(0, cy - roi_height // 2)
    x_end = min(gray_image.shape[1], x + roi_width)  # Ensure ROI is within image bounds
    y_end = min(gray_image.shape[0], y + roi_height)  # Ensure ROI is within image bounds

    roi = gray_image[y:y_end, x:x_end]

    return roi


def resize_roi(roi):
    return cv2.resize(roi, TARGET_SIZE)


def normalize_roi(roi):
    return roi.astype(np.float32) / 255.0


def preprocess_image(image_path):
    try:
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Error: Could not read image at {image_path}")
            return None

        gray_image = to_grayscale(original_image)
        enhanced_gray_image = enhance_contrast_clahe(gray_image)
        enhanced_gray_image = reduce_noise(enhanced_gray_image)

        binary_image, contours = segment_palm(enhanced_gray_image)
        roi = extract_roi_centroid(enhanced_gray_image, contours)

        if roi is None or roi.size == 0:
            print(f"Error: ROI extraction failed for {image_path} or ROI is empty.")
            return None

        resized_roi = resize_roi(roi)
        normalized_roi = normalize_roi(resized_roi)

        return normalized_roi

    except Exception as e:
        print(f"Error during preprocessing of {image_path}: {e}")
        return None

# normalized_roi = preprocess_image("../dataset/archive/session1/00001.tiff")
# cv2.imshow('Final Preprocess Image', normalized_roi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
