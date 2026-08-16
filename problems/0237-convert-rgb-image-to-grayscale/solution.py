import numpy as np

def rgb_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminosity method.
    
    Args:
        image: RGB image as list or numpy array of shape (H, W, 3)
               with values in range [0, 255]
    
    Returns:
        Grayscale image as 2D list with integer values,
        or -1 if input is invalid
    """
    if len(image) == 0 or len(image[0]) == 0:
        return -1

    if not isinstance(image[0][0], (list, np.ndarray)) or len(image[0][0]) != 3:
        return -1

    ans = []
    h = len(image)
    w = len(image[0])

    for i in range(h):
        temp = []
        for j in range(w):
            if(image[i][j][0]<256 and image[i][j][1]<256 and image[i][j][2]<256):
                val = int(np.round(
                    .299 * image[i][j][0] +
                    .587 * image[i][j][1] +
                    .114 * image[i][j][2]
                ))
                temp.append(val)
            else:
                return -1
        ans.append(temp)

    return ans