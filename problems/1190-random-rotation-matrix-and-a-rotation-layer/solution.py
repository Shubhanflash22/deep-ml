import math
def rotation_layer(X, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    rotated_points = []
    for x, y in X:
        x_new = x * cos_a - y * sin_a
        y_new = x * sin_a + y * cos_a
        rotated_points.append([round(x_new, 10), round(y_new, 10)])
    return rotated_points 