"""
Functions for creating graph nodes from individual pixels.
"""

import numpy as np

def pixel_nodes(image, downsample_factor=1):
    """
    Creates graph nodes from individual pixels, with optional downsampling.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape (H, W, C)
    downsample_factor : int, optional
        Factor by which to downsample the image, by default 1 (no downsampling)
        
    Returns
    -------
    dict
        Dictionary containing node information:
        - 'pixels' : downsampled image
        - 'centroids' : node centroid coordinates (N, 2)
        - 'positions' : grid positions of each pixel (N, 2) as (row_idx, col_idx)
        - 'values' : pixel values (N, C)
    """
    # Downsample image if needed
    if downsample_factor > 1:
        height, width = image.shape[:2]
        new_height = height // downsample_factor
        new_width = width // downsample_factor
        
        # Simple downsampling by taking strided pixels
        # For more sophisticated downsampling, consider using cv2.resize or skimage.transform.resize
        downsampled = image[::downsample_factor, ::downsample_factor]
    else:
        downsampled = image
    
    # Get dimensions of downsampled image
    height, width = downsampled.shape[:2]
    
    # Create pixel positions and centroids
    positions = []
    centroids = []
    values = []
    
    for i in range(height):
        for j in range(width):
            positions.append((i, j))
            # Centroids are shifted by 0.5 to be at the center of the pixel
            centroids.append((i + 0.5, j + 0.5))
            values.append(downsampled[i, j])
    
    return {
        'pixels': downsampled,
        'centroids': np.array(centroids),
        'positions': np.array(positions),
        'values': np.array(values)
    }

def keypoint_nodes(image, detector='sift', max_keypoints=100):
    """
    Creates graph nodes from keypoints detected in the image.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape (H, W, C)
    detector : str, optional
        Keypoint detector to use ('sift', 'orb', 'fast'), by default 'sift'
    max_keypoints : int, optional
        Maximum number of keypoints to return, by default 100
        
    Returns
    -------
    dict
        Dictionary containing node information:
        - 'keypoints' : list of cv2.KeyPoint objects
        - 'centroids' : node centroid coordinates (N, 2)
        - 'descriptors' : keypoint descriptors (N, D)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV (cv2) is required for keypoint detection.")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] > 1:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Ensure image is in the right format
    if gray.dtype != np.uint8:
        gray = (gray * 255).astype(np.uint8)
    
    # Initialize detector
    if detector.lower() == 'sift':
        det = cv2.SIFT_create(nfeatures=max_keypoints)
    elif detector.lower() == 'orb':
        det = cv2.ORB_create(nfeatures=max_keypoints)
    elif detector.lower() == 'fast':
        # FAST doesn't compute descriptors
        fast = cv2.FastFeatureDetector_create()
        keypoints = fast.detect(gray, None)
        # Use BRIEF for descriptors
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        keypoints, descriptors = brief.compute(gray, keypoints)
        
        # Limit to max_keypoints
        if len(keypoints) > max_keypoints:
            keypoints = keypoints[:max_keypoints]
            descriptors = descriptors[:max_keypoints]
        
        # Extract centroids
        centroids = np.array([(kp.pt[1], kp.pt[0]) for kp in keypoints])  # (y, x) format
        
        return {
            'keypoints': keypoints,
            'centroids': centroids,
            'descriptors': descriptors
        }
    else:
        raise ValueError(f"Unsupported detector: {detector}. Use 'sift', 'orb', or 'fast'.")
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = det.detectAndCompute(gray, None)
    
    # Limit to max_keypoints
    if len(keypoints) > max_keypoints:
        keypoints = keypoints[:max_keypoints]
        descriptors = descriptors[:max_keypoints]
    
    # Extract centroids
    centroids = np.array([(kp.pt[1], kp.pt[0]) for kp in keypoints])  # (y, x) format
    
    return {
        'keypoints': keypoints,
        'centroids': centroids,
        'descriptors': descriptors
    }