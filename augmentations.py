"""
augmentations.py

SEM (Scanning Electron Microscopy) noise and geometric augmentation pipeline for the
Applied Materials Drift-Sense challenge.

Includes:
- Independent Gaussian noise & Poisson noise
- Gaussian blur
- Edge brightening (SEM secondary electron charging effect)
- Scan-line raster artifacts
- Brightness & contrast variations
- Geometric warp (rotation +/-3 deg, scale 0.95-1.05, translation +/-10 px) with exact coordinate tracking
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Any


def apply_edge_brightening(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Simulates secondary electron accumulation along steep semiconductor feature edges
    using Sobel spatial gradient magnitude enhancement.

    Args:
        image (np.ndarray): Input grayscale image (float32).
        strength (float): Edge enhancement factor.

    Returns:
        np.ndarray: Edge-enhanced image (float32).
    """
    if strength <= 0:
        return image.copy()
        
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    
    # Normalize gradient magnitude
    max_val = grad_mag.max()
    if max_val > 0:
        grad_mag = grad_mag / max_val
        
    brightened = image + (grad_mag * strength * 75.0)
    return np.clip(brightened, 0.0, 255.0)


def add_gaussian_noise(image: np.ndarray, std: float, rng: np.random.RandomState) -> np.ndarray:
    """
    Applies independent additive Gaussian sensor noise.

    Args:
        image (np.ndarray): Input float32 grayscale image.
        std (float): Standard deviation of Gaussian noise.
        rng (np.random.RandomState): Random state.

    Returns:
        np.ndarray: Noisy image (float32).
    """
    if std <= 0:
        return image.copy()
    noise = rng.normal(loc=0.0, scale=std, size=image.shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 255.0)


def add_poisson_noise(image: np.ndarray, peak: float, rng: np.random.RandomState) -> np.ndarray:
    """
    Applies Poisson shot noise simulating electron count fluctuations in SEM inspection.

    Args:
        image (np.ndarray): Input float32 grayscale image [0, 255].
        peak (float): Peak electron count scaling factor (higher means lower noise).
        rng (np.random.RandomState): Random state.

    Returns:
        np.ndarray: Poisson noise augmented image (float32).
    """
    if peak <= 0:
        return image.copy()
    # Normalize image to [0, 1] then scale by peak
    norm_img = np.maximum(0.0, image) / 255.0
    scaled = norm_img * peak
    # Generate Poisson values
    noisy_scaled = rng.poisson(scaled).astype(np.float32)
    # Rescale back to [0, 255]
    rescaled = (noisy_scaled / peak) * 255.0
    return np.clip(rescaled, 0.0, 255.0)


def add_scanline_artifacts(image: np.ndarray, strength: float, rng: np.random.RandomState) -> np.ndarray:
    """
    Simulates SEM raster scan-line intensity fluctuations across horizontal rows.

    Args:
        image (np.ndarray): Input float32 grayscale image.
        strength (float): Scanline noise amplitude.
        rng (np.random.RandomState): Random state.

    Returns:
        np.ndarray: Image with scanline artifacts (float32).
    """
    if strength <= 0:
        return image.copy()
    h, w = image.shape
    # Generate line-by-line brightness offsets
    row_offsets = rng.normal(loc=0.0, scale=strength, size=(h, 1)).astype(np.float32)
    return np.clip(image + row_offsets, 0.0, 255.0)


def apply_brightness_contrast(
    image: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0
) -> np.ndarray:
    """
    Applies contrast adjustment (alpha) and brightness shift (beta).

    Args:
        image (np.ndarray): Input float32 grayscale image.
        alpha (float): Contrast multiplier (e.g. 0.8 - 1.2).
        beta (float): Brightness shift (e.g. -25 - 25).

    Returns:
        np.ndarray: Contrast & brightness adjusted image.
    """
    adjusted = alpha * image + beta
    return np.clip(adjusted, 0.0, 255.0)


def apply_gaussian_blur(image: np.ndarray, kernel_size: int, sigma: float = 0.0) -> np.ndarray:
    """
    Applies Gaussian blur for optical/focus degradation.

    Args:
        image (np.ndarray): Input float32 grayscale image.
        kernel_size (int): Kernel size (must be odd integer >= 3).
        sigma (float): Blur sigma.

    Returns:
        np.ndarray: Blurred image.
    """
    if kernel_size < 3 or kernel_size % 2 == 0:
        return image.copy()
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)


def apply_geometric_transform(
    image: np.ndarray,
    angle: float,
    scale: float,
    dx: float,
    dy: float,
    center: Tuple[float, float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies spatial rotation, scaling, and translation via affine transformation matrix.

    Args:
        image (np.ndarray): Input image.
        angle (float): Rotation angle in degrees (e.g. -3.0 to +3.0).
        scale (float): Scale factor (e.g. 0.95 to 1.05).
        dx (float): X translation in pixels.
        dy (float): Y translation in pixels.
        center (Tuple[float, float], optional): Rotation center (default image center).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Transformed image.
            - 2x3 Affine transformation matrix M.
    """
    h, w = image.shape[:2]
    if center is None:
        center = (w / 2.0, h / 2.0)
        
    # Get 2x3 affine rotation & scale matrix around image center
    M = cv2.getRotationMatrix2D(center, angle, scale)
    # Add translation offsets
    M[0, 2] += dx
    M[1, 2] += dy
    
    # Warp image with reflection padding to prevent black borders
    warped = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return warped, M


def transform_coordinate(coord: Tuple[float, float], M: np.ndarray) -> Tuple[float, float]:
    """
    Transforms a single (x, y) point using an affine transformation matrix M.

    Args:
        coord (Tuple[float, float]): Input (x, y) coordinates.
        M (np.ndarray): 2x3 Affine transformation matrix.

    Returns:
        Tuple[float, float]: Updated (x, y) coordinates in transformed space.
    """
    x, y = coord
    x_new = M[0, 0] * x + M[0, 1] * y + M[0, 2]
    y_new = M[1, 0] * x + M[1, 1] * y + M[1, 2]
    return (float(x_new), float(y_new))


def apply_full_augmentation_pipeline(
    image: np.ndarray,
    rng: np.random.RandomState,
    center_coord: Tuple[float, float] = None,
    is_search: bool = False
) -> Tuple[np.ndarray, Tuple[float, float], Dict[str, Any]]:
    """
    Master SEM augmentation pipeline applying independent physical, sensor,
    and geometric degradations.

    Args:
        image (np.ndarray): Input float32 1000x1000 image.
        rng (np.random.RandomState): Independent random generator.
        center_coord (Tuple[float, float], optional): Ground-truth center (x, y) if tracking.
        is_search (bool): If True, applies geometric warp and tracks center coordinate.

    Returns:
        Tuple[np.ndarray, Tuple[float, float], Dict[str, Any]]:
            - Augmented 1000x1000 image (float32).
            - Updated center coordinate (x, y) (or unchanged if center_coord is None).
            - Dictionary of randomized augmentation parameter settings.
    """
    augmented = image.copy()
    
    # 1. Randomize Augmentation Parameters
    params = {
        'edge_brightening_strength': float(rng.uniform(0.2, 0.8)),
        'contrast_scale': float(rng.uniform(0.85, 1.15)),
        'brightness_shift': float(rng.uniform(-20.0, 20.0)),
        'gaussian_noise_std': float(rng.uniform(8.0, 25.0 if is_search else 18.0)),
        'poisson_peak': float(rng.uniform(100.0, 500.0)),
        'blur_kernel': int(rng.choice([1, 3, 5])),
        'scanline_strength': float(rng.uniform(1.0, 6.0)),
        'angle': float(rng.uniform(-3.0, 3.0)),
        'scale': float(rng.uniform(0.95, 1.05)),
        'dx': float(rng.uniform(-8.0, 8.0)),
        'dy': float(rng.uniform(-8.0, 8.0)),
    }

    # 2. Edge Brightening (Secondary Electron Effect)
    augmented = apply_edge_brightening(augmented, params['edge_brightening_strength'])

    # 3. Brightness & Contrast Adjustment
    augmented = apply_brightness_contrast(augmented, params['contrast_scale'], params['brightness_shift'])

    # 4. Optional Gaussian Blur
    if params['blur_kernel'] >= 3:
        augmented = apply_gaussian_blur(augmented, params['blur_kernel'], sigma=0.8)

    # 5. Add Poisson Noise & Gaussian Sensor Noise
    augmented = add_poisson_noise(augmented, params['poisson_peak'], rng)
    augmented = add_gaussian_noise(augmented, params['gaussian_noise_std'], rng)

    # 6. Add Scanline Artifacts
    augmented = add_scanline_artifacts(augmented, params['scanline_strength'], rng)

    # 7. Geometric Augmentation (Rotation, Scale, Translation)
    updated_coord = center_coord
    if is_search:
        augmented, M = apply_geometric_transform(
            augmented,
            angle=params['angle'],
            scale=params['scale'],
            dx=params['dx'],
            dy=params['dy']
        )
        if center_coord is not None:
            updated_coord = transform_coordinate(center_coord, M)
    else:
        # Reference image can also undergo slight geometric warp
        augmented, _ = apply_geometric_transform(
            augmented,
            angle=params['angle'],
            scale=params['scale'],
            dx=params['dx'],
            dy=params['dy']
        )

    return augmented, updated_coord, params
