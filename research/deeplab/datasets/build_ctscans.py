import numpy as np
import os
import tensorflow as tf
import glob
from PIL import Image
import pdb as bug
from tqdm import tqdm, trange
import sys
import build_data
import math


FLAGS = tf.app.flags.FLAGS
FORMAT = 'png'
IMG_DIRNAME = 'images_pngs'
MASK_DIRNAME = 'masks_pngs'


def remove_color_map(input_dir):
    #if not tf.gfile.IsDirectory(output_dir):
    #    tf.gfile.MakeDirs(output_dir)

    filenames = glob.glob(os.path.join(input_dir, '*/*/*.' + FORMAT))
    for filename in tqdm(filenames):
        grey_img = Image.open(filename).convert('L')
        #filename = os.path.join(output_dir, *(annotation.split('/')[-2:]))
        #headname = os.path.split(filename)[0]
        #if not tf.gfile.IsDirectory(headname):
        #    tf.gfile.MakeDirs(headname)
        #pil_image = Image.fromarray(raw_annotation.astype(dtype=np.uint8))
        with tf.gfile.Open(filename, mode='w') as f:
            grey_img.save(f, 'PNG')


def convert_dataset(dir, output_dir, num_shards=7, seed=123):
    np.random.seed(seed)
    reader = build_data.ImageReader('png', channels=1)
    dataset = os.path.basename(dir)
    if not tf.gfile.IsDirectory(output_dir):
        tf.gfile.MakeDirs(output_dir)

    sys.stdout.write('Processing ' + dir)

    # Getting filenames
    img_filenames = glob.glob(os.path.join(dir, IMG_DIRNAME, '*', '*.png'))
    seg_filenames = glob.glob(os.path.join(dir, MASK_DIRNAME, '*', '*.png'))
    num_images = len(img_filenames)
    num_per_shard = int(math.ceil(num_images / float(num_shards)))

    # Shuffle data
    z = list(zip(img_filenames, seg_filenames))
    np.random.shuffle(z)
    img_filenames[:], seg_filenames[:] = zip(*z)

    for shard in trange(num_shards):
        output_filename = os.path.join(
            output_dir,
            '%s-%03d-of-%03d.tfrecord' % (dataset, shard, num_shards))
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard * num_per_shard
            end_idx = min((shard + 1) * num_per_shard, num_images)
            for img_filename, seg_filename in list(zip(img_filenames, seg_filenames))[start_idx:end_idx]:
                img = tf.gfile.Open(img_filename, 'rb').read()
                height, width = reader.read_image_dims(img)
                # todo: check if height and width match between image and label
                seg = tf.gfile.Open(seg_filename, 'rb').read()
                example = build_data.image_seg_to_tfexample(
                    img, img_filename, height, width, seg)
                tfrecord_writer.write(example.SerializeToString())


def main(unused_argv):
    for input_dir in glob.glob('ctscans/*'):
        # masks_dir = os.path.join(dir, MASK_DIRNAME, '*')
        # train_output_dir = os.path.join(dir, SEG_DIRNAME)
        print('removing colormap for images in {}'.format(input_dir))
        remove_color_map(input_dir)

        print('converting images to tfrecord in {}'.format(input_dir))
        convert_dataset(input_dir, 'ctscans/tfrecord')


if __name__ == '__main__':
    tf.app.run()
