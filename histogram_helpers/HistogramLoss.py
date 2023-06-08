import torch
import torch.nn as nn
import numpy as np

## calculate color histogram of two images with numpy images are already rgb
def color_histogram_of_training_image(image):
    r_channel = torch.zeros(16,256)
    g_channel = torch.zeros(16,256)
    b_channel = torch.zeros(16,256)
    print("started loss")

    for i in range(image.shape[2]):
        for j in range(image.shape[3]):
            for z in range(16):
                r_channel[z, int(((image[z, 0,i,j].cpu().item() + 1 ) / 2) * 255)] += 1
                g_channel[z,int(((image[z, 1,i,j].cpu().item() + 1 ) / 2) * 255)] += 1
                b_channel[z, int(((image[z, 2,i,j].cpu().item() + 1 ) / 2) * 255)] += 1
                
    print("ended loss")

    return r_channel, g_channel, b_channel

#calculate color historgram difference of two images
def color_histogram_distance_of_two_images(image1, image2):
    r1, g1, b1 = color_histogram_of_training_image(image1)
    r2, g2, b2 = color_histogram_of_training_image(image2)

    r = np.sum(np.sqrt(np.sum((r1 - r2)**2)))
    g = np.sum(np.sqrt(np.sum((g1 - g2)**2)))
    b = np.sum(np.sqrt(np.sum((b1 - b2)**2)))

    return r, g, b