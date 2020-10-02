import sys
import os

import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# This line is only required if you want to try the lib without installing it
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from tf_netbuilder.models.openpose_singlenet import create_openpose_singlenet


def create_figure(outputs, show_paf_idx, show_heatmap_idx):

    paf1 = outputs[0]
    paf2 = outputs[1]
    paf3 = outputs[2]
    heatmap1 = outputs[3]

    figure = plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1, title='stage 1 - paf')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(paf1[0, :, :, show_paf_idx], cmap='gray')

    plt.subplot(2, 2, 2, title='stage 2 - paf')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(paf2[0, :, :, show_paf_idx], cmap='gray')

    plt.subplot(2, 2, 3, title='stage 3 - paf')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(paf3[0, :, :, show_paf_idx], cmap='gray')

    plt.subplot(2, 2, 4, title='stage 4 - heatmaps')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(heatmap1[0, :, :, show_heatmap_idx], cmap='gray')

    return figure


def evaluate(test_img_path: str, output_img_path):

    model = create_openpose_singlenet(pretrained=True)

    image = cv2.imread(test_img_path)
    im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_rgb = cv2.resize(im_rgb, (224, 224))
    input_img = im_rgb[np.newaxis, :, :, :]
    inputs = [tf.convert_to_tensor(input_img)]

    outputs = model.predict(inputs)

    fig = create_figure(outputs, show_paf_idx=0, show_heatmap_idx=0)

    fig.savefig(output_img_path)


if __name__ == '__main__':

    evaluate(test_img_path='../resources/ski_224.jpg',
             output_img_path='openpose_singlenet_results.png')

    print("Results saved in openpose_singlenet_results.png")
