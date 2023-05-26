import img_manipulations
import random


def generate_spoofed_dataset(dataset):
    spoofed_dataset = np.array([])

    #distributions of filters (relative values)
    rotation_odd = 1
    gaussian_blur_odd = 1
    box_blur_odd = 1
    uniform_noise_odd = 1
    perlin_noise_odd = 1
    flip_image_odd = 1

    total = rotation_odd + gaussian_blur_odd + box_blur_odd + \
            uniform_noise_odd + perlin_noise_odd + flip_image_odd

    for image in dataset:
        rand = random.random() * total



# fig = plt.figure(figsize=(5, 5))

# for i in range(5):
#     for j in range(5):
#         fig.add_subplot(i, j)
#         plt.imshow(gaussian_blur(x_train[j], i))
# plt.show()

# f, axes = plt.subplots(5, 5)
    
# for i, row in enumerate(axes):
#     for j, ax in enumerate(row):
#         ax.imshow(gaussian_blur(x_train[j], 2))

# plt.show()