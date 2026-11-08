import numpy as np

def flip_image(image, direction):
    """
    Flip an image horizontally or vertically.

    Args:
        image: 2D or 3D list/array representing a grayscale or RGB image
        direction: string, either 'horizontal' or 'vertical'

    Returns:
        Flipped image as a nested list, or -1 if input is invalid
    """
    # Convert input to a NumPy array
    image = np.array(image)

    # Validate input dimensions
    if image.ndim not in (2, 3):
        return -1

    # Flip the image
    if direction == 'horizontal':
        flipped = np.fliplr(image)
    elif direction == 'vertical':
        flipped = np.flipud(image)
    else:
        return -1

    # Return as a nested list
    return flipped.tolist()