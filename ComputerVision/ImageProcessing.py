import numpy as np
import matplotlib.pyplot as plt

''' Image Processing '''

image = plt.imread('/Users/fer/Downloads/Jacobo.png')
# image.setflags(write=1)
# image.flags.writeable = True
image_plot = 0
if image_plot == 1:
    plt.imshow(image)
    plt.show(), plt.clf()

print(image.shape) # dimensions
print('Jacobo pixel data: \n', image)
print('\n\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n\n')
print('1st list: \n', image[1])
print('\nThis is the width of the list: {:.0f} pixels'.format(len(image[0]))) # width
print('this is the height of the list: {:.0f} pixels'.format(len(image))) # height
print('This is Jacobo´s head: \n', image[730:914, 497:660, :])
image_head = image[730:914, 497:660, :]

image_head_plot = 0
if image_head_plot == 1:
    plt.imshow(image_head)
    plt.show(), plt.clf()

color_transformation = 1
if color_transformation == 1:
    image[730:914, 497:660, :] = [0, 0, 1, 1]
    plt.imshow(image)
    plt.show(), plt.clf()

color_transformation_advanced = 0
if color_transformation_advanced == 1:
    image[:, :, 1] = 0
    # jacobo[:, :, 2] =
    image[:, :, 3] = 0
    plt.imshow(image)
    plt.show(), plt.clf()

print(image[0][0][:3])

''' Video Processing '''
