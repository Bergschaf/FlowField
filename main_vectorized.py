import numpy
import numpy as np
from PIL import Image, ImageFilter
from perlin_noise import PerlinNoise
import matplotlib.pyplot as plt

noise = PerlinNoise(octaves=10, seed=5)
size = 256
vector_field = np.zeros((size, size, 2), dtype=np.float32)
for i in range(vector_field.shape[0]):
    for j in range(vector_field.shape[1]):
        factor = size * 2
        vector_field[i, j] = [noise([i / factor, j / factor]), noise([i / factor, j / factor, 1])]

# 0.2 should be added to all positive values in the vector field
# 0.2 should be subtracted from all negative values in the vector field

vector_field[vector_field >= 0] += vector_field[vector_field >= 0] * 0.3
vector_field[vector_field < 0] -= vector_field[vector_field < 0] * 0.3

timesteps = 10000
timestep_size = 0.05
num_points = 5000

points = np.zeros((timesteps, num_points, 2), dtype=np.float32)
points_acceleration = np.zeros((num_points, 2), dtype=np.float32)
points[0] = np.random.uniform(0, 1, (num_points, 2)) * size

for i in range(1, timesteps):
    int_pos = points[i - 1].round().astype(np.int32)
    int_pos.clip(0, size - 1, out=int_pos)
    points_acceleration = vector_field[int_pos[:, 0], int_pos[:, 1]]
    points[i] = points[i - 1] + points_acceleration * timestep_size
    if i % 100 == 0:
        print(i)

image_size_factor = 16
size *= image_size_factor
img = np.zeros((size, size, 3), dtype=np.uint16)
for i in range(1, timesteps):
    int_pos = (points[i - 1] * image_size_factor).round().astype(np.int32)
    int_pos.clip(0, size - 1, out=int_pos)
    color = np.array([255 - int(i / timesteps * 255), 0, int(i / timesteps * 255)], dtype=np.uint16)
    # assign the color to the image if the pixel is not already colored
    img[int_pos[:, 0], int_pos[:, 1]] += color

    if i % 100 == 0:
        print(i)

img.clip(0, 255, out=img)
target_ratio = 2560 / 1440
img = img[0:int(size / target_ratio), :, :]
img = img.astype(np.uint8)
image = Image.fromarray(img)
# apply convolution to image
# image = image.filter(ImageFilter.GaussianBlur(radius=1))
# target size: 2560x1440

image.thumbnail((2560, 1440), Image.Resampling.LANCZOS)
image.save("1.png")

"""image_factor = 4
image = Image.new("RGB", (size * image_factor, size * image_factor))
for i in range(1,timesteps):
    for j in range(num_points):
        pos = [int(points[i, j, 0].__round__()), int(points[i, j, 1].__round__())]
        if 0 > pos[0] or pos[0] > size - 1:
            pos[0] = 0 if pos[0] < 0 else size - 1
        if 0 > pos[1] or pos[1] > size - 1:
            pos[1] = 0 if pos[1] < 0 else size - 1

        prev_pos = [int(points[i - 1, j, 0].__round__()), int(points[i - 1, j, 1].__round__())]
        if 0 > prev_pos[0] or prev_pos[0] > size - 1:
            prev_pos[0] = 0 if prev_pos[0] < 0 else size - 1
        if 0 > prev_pos[1] or prev_pos[1] > size - 1:
            prev_pos[1] = 0 if prev_pos[1] < 0 else size - 1

        # draw line from prev_pos to pos (dda algorithm)
        # multiply pos by image_factor
        pos = [pos[0] * image_factor, pos[1] * image_factor]
        prev_pos = [prev_pos[0] * image_factor, prev_pos[1] * image_factor]

        dx = pos[0] - prev_pos[0]
        dy = pos[1] - prev_pos[1]
        steps = abs(dx) if abs(dx) > abs(dy) else abs(dy)
        if steps == 0:
            continue
        x_inc = dx / steps
        y_inc = dy / steps
        x = prev_pos[0]
        y = prev_pos[1]
        for k in range(steps):
            image.putpixel((int(x), int(y)), (255, 255, 255))
            x += x_inc
            y += y_inc
    print(i)"""

# image.save(f"{1}.png")
