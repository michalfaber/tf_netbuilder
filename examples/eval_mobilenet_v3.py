import sys
import os
import cv2
import numpy as np
import tensorflow as tf

# This line is only required if you want to try the lib without installing it
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from tf_netbuilder.models.mobilenet_v3 import create_mobilenet_v3_224_1x


def create_readable_names_for_imagenet_labels():

    synset_url = '../resources/imagenet_lsvrc_2015_synsets.txt'
    synset_to_human_url = '../resources/imagenet_metadata.txt'

    synset_list = [s.strip() for s in open(synset_url).readlines()]
    num_synsets_in_ilsvrc = len(synset_list)
    assert num_synsets_in_ilsvrc == 1000

    synset_to_human_list = open(synset_to_human_url).readlines()
    num_synsets_in_all_imagenet = len(synset_to_human_list)
    assert num_synsets_in_all_imagenet == 21842

    synset_to_human = {}
    for s in synset_to_human_list:
        parts = s.strip().split('\t')
        assert len(parts) == 2
        synset = parts[0]
        human = parts[1]
        synset_to_human[synset] = human

    label_index = 1
    labels_to_names = {0: 'background'}
    for synset in synset_list:
        name = synset_to_human[synset]
        labels_to_names[label_index] = name
        label_index += 1

    return labels_to_names


def evaluate(test_image_path: str, imagenet_label_map: map):

    model = create_mobilenet_v3_224_1x(pretrained=True)

    image = cv2.imread(test_image_path)
    im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_rgb = cv2.resize(im_rgb, (224, 224))
    input_img = im_rgb[np.newaxis, :, :, :]
    inputs = [tf.convert_to_tensor(input_img)]

    logits, preds = model.predict(inputs, steps=1)

    print("File: ", test_image_path, ", Top 1 prediction: ", preds.argmax(),
          imagenet_label_map[preds.argmax()], preds.max())


if __name__ == '__main__':
    label_map = create_readable_names_for_imagenet_labels()

    evaluate(test_image_path='../resources/dog.jpg', imagenet_label_map=label_map)
    evaluate(test_image_path='../resources/panda.jpg', imagenet_label_map=label_map)

