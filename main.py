import numpy as np
from PIL import Image
from perlin_noise import PerlinNoise
import matplotlib.pyplot as plt

noise = PerlinNoise(octaves=10, seed=2)
size = 256
vector_field = np.zeros((size, size, 2), dtype=np.float32)
for i in range(vector_field.shape[0]):
    for j in range(vector_field.shape[1]):
        vector_field[i, j] = [noise([i / size, j / size]) + 0.5, noise([i / size, j / size, 1]) + 0.5]
timesteps = 10000
timestep_size = 0.4
num_points = 1000

points = np.zeros((timesteps, num_points, 2), dtype=np.float32)
points_acceleration = np.zeros((num_points, 2), dtype=np.float32)
points[0] = np.random.uniform(0, 1, (num_points, 2)) * size

for i in range(1, timesteps):
    for j in range(num_points):
        pos = [int(points[i - 1, j, 0].__round__()), int(points[i - 1, j, 1].__round__())]
        if 0 > pos[0] or pos[0] > size - 1:
            pos[0] = 0 if pos[0] < 0 else size - 1
        if 0 > pos[1] or pos[1] > size - 1:
            pos[1] = 0 if pos[1] < 0 else size - 1

        vector = vector_field[pos[0], pos[1]]
        points_acceleration[j] = vector
        points[i, j] = points[i - 1, j] + points_acceleration[j] * timestep_size
    print(i)


image_size_factor = 4
size *= image_size_factor
image = Image.new("RGB", (size, size))
for i in range(1, timesteps):
    for j in range(num_points):
        pos = [int((points[i - 1, j, 0] * image_size_factor).__round__()),
               int((points[i - 1, j, 1] * image_size_factor).__round__())]
        if 0 > pos[0] or pos[0] > size - 1:
            pos[0] = 0 if pos[0] < 0 else size - 1
        if 0 > pos[1] or pos[1] > size - 1:
            pos[1] = 0 if pos[1] < 0 else size - 1
        pos = (pos[0], pos[1])

        image.putpixel(pos, (255, 255, 255))
    print(i)
image.save(f"{1}.png")

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

image.save(f"{1}.png")
