##################################################
# Author: {Christian F. Baumgartner} (c.f.baumgartner@gmail.com)
# https://github.com/baumgach/vagan-code
##################################################
from skimage import filters
import logging
import h5py
import os.path
import utils as utils
import os
import torch.utils.data as data
import torch
import numpy as np
from torch.utils.data import Dataset


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


EFFECT_SIZE = 100
NUM_SAMPLES = 20000
MOVING_EFFECT = True
RESCALE_TO_ONE = True


CACHE = {}

DSET_SPLIT_SIZES = {
    'train': [0, 0.8**2],
    'val': [0.8**2, 0.8],
    'test': [0.8, 1],
    'pred': [0, 1],
    None: [0, 1],
}


class SynthDataset(Dataset):
    '''
    Subtype of torch.utils.data.Dataset.
    for more info: http://pytorch.org/docs/master/data.html
    This class use the (copied) cope from the reference paper's official repo
    https://github.com/baumgach/vagan-code
    '''

    def __init__(self, opt, anomaly, mode='train', transform=None):
        super(SynthDataset, self).__init__()
        self.transform = transform
        if 'loaded' not in CACHE:
            self.load_cache(opt)
        split_size = DSET_SPLIT_SIZES[mode]
        idxs = np.where(CACHE['y'] == int(anomaly))[0]
        l1 = int(len(idxs) * split_size[0])
        l2 = int(len(idxs) * split_size[1])
        self.idxs = idxs[l1:l2]

    def load_cache(self, opt):
        data = load_and_maybe_generate_data(output_folder=opt.dataroot,
                                                                  image_size=opt.image_size,
                                                                  force_overwrite=False)

        lhr_size = data['features'].shape[0]
        imsize = int(np.sqrt(lhr_size))

        images = np.reshape(data['features'][:], [imsize, imsize, -1])
        images = np.transpose(images, [2, 0, 1])

        masks = np.reshape(data['gt'][:], [imsize, imsize, -1])
        masks = np.transpose(masks, [2, 0, 1])

        labels = data['labels'][:]

        CACHE['X'] = images
        CACHE['y'] = labels
        CACHE['masks'] = masks
        CACHE['loaded'] = True

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        idx = self.idxs[idx]
        x = CACHE['X'][idx]
        y = CACHE['y'][idx]
        mask = CACHE['masks'][idx]
        c_org = np.zeros((2,))
        c_org[y] = 1
        x = np.expand_dims(x, 0)

        if self.transform:
            x = self.transform(x)
        return x, c_org, mask


def load_and_maybe_generate_data(output_folder,
                                 image_size,
                                 effect_size=EFFECT_SIZE,
                                 num_samples=NUM_SAMPLES,
                                 moving_effect=MOVING_EFFECT,
                                 scale_to_one=RESCALE_TO_ONE,
                                 force_overwrite=False):

    size_str = str(image_size)
    effect_str = str(effect_size)
    sample_str = str(num_samples)

    rescale_postfix = '_intrangeone' if scale_to_one else ''
    moving_postfix = '_moving' if moving_effect else ''

    data_file_name = 'synthdata_num_%s_imsize_%s_effect_%s%s%s.hdf5' % \
                     (sample_str, size_str, effect_str,
                      moving_postfix, rescale_postfix)
    data_file_path = os.path.join(output_folder, data_file_name)

    utils.makefolder(output_folder)

    if not os.path.exists(data_file_path) or force_overwrite:
        logging.info(
            'This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(data_file_path,
                     effect_size,
                     num_samples,
                     image_size,
                     moving_effect,
                     scale_to_one,
                     save_type='hdf5')
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')


def prepare_data(out_path,
                 effect_size=50.,
                 num_samples=100,
                 image_size=100,
                 moving_effect=True,
                 scale_intensities_to_one=True,
                 save_type='hdf5'):

    # Constants
    stdbckg = 50.  # std deviation of the background
    stdkernel = 2.5  # std deviation of the Gaussian smoothing kernel
    block1size = 10  # size of the first block
    block2size = 10  # size of the 2nd block

    offset = int((image_size / 3.5) + 0.5)
    block2offset_ = np.asarray([offset, offset])
    block3size = 10  # size of the 3rd block
    block3offset_ = np.asarray([-offset, -offset])
    norm_percentile = 0

    numNsamples = num_samples // 2
    numP1samples = np.int(num_samples // 4)
    numP2samples = np.int(num_samples // 4)

    Features = np.zeros(
        [image_size ** 2, numNsamples + numP1samples + numP2samples])
    GT = np.zeros([image_size ** 2, numNsamples + numP1samples + numP2samples])
    Labels = np.zeros(numNsamples+numP1samples+numP2samples)
    half_imsize = np.int(image_size / 2)

    # Generate images of class 1 with subtype A (box in the centre and upper left)
    for n in range(numP1samples):

        I = np.zeros([image_size, image_size])
        I[half_imsize - block1size: half_imsize + block1size, half_imsize -
            block1size: half_imsize + block1size] = effect_size

        if moving_effect:
            block2offset = block2offset_ + np.random.randint(-5, 5, size=2)
        else:
            block2offset = block2offset_

        I[half_imsize + block2offset[0] - block2size: half_imsize + block2offset[0] + block2size,
          half_imsize + block2offset[1] - block2size: half_imsize + block2offset[1] + block2size] = effect_size

        GT[:, n] = I.reshape(image_size ** 2) > 0
        noise = np.random.normal(
            scale=stdbckg, size=np.asarray([image_size, image_size]))
        smnoise = filters.gaussian(noise, stdkernel)
        smnoise = smnoise / np.std(smnoise) * stdbckg
        J = I + smnoise

        if scale_intensities_to_one:
            J = utils.map_image_to_intensity_range(
                J, -1, 1, percentiles=norm_percentile)

        Features[:, n] = J.reshape(image_size ** 2)
        Labels[n] = 1

    # Generate images of class 1 with subtype B (box in the centre and lower right)
    for n in range(numP2samples):

        I = np.zeros([image_size, image_size])
        I[half_imsize - block1size: half_imsize + block1size,
          half_imsize - block1size: half_imsize + block1size] = effect_size

        if moving_effect:
            block3offset = block3offset_ + np.random.randint(-5, 5, size=2)
        else:
            block3offset = block3offset_

        I[half_imsize + block3offset[0] - block3size: half_imsize + block3offset[0] + block3size,
          half_imsize + block3offset[1] - block3size: half_imsize + block3offset[1] + block3size] = effect_size

        GT[:, n+numP1samples] = I.reshape(image_size ** 2) > 0
        noise = np.random.normal(
            scale=stdbckg, size=np.asarray([image_size, image_size]))
        smnoise = filters.gaussian(noise, stdkernel)
        smnoise = smnoise / np.std(smnoise) * stdbckg
        J = I + smnoise

        if scale_intensities_to_one:
            J = utils.map_image_to_intensity_range(
                J, -1, 1, percentiles=norm_percentile)

        Features[:, n+numP1samples] = J.reshape(image_size ** 2)
        Labels[n+numP1samples] = 1

    # Generate image of class 0 (only noise)
    for n in range(numNsamples):

        I = np.zeros([image_size, image_size])
        noise = np.random.normal(
            scale=stdbckg, size=np.asarray([image_size, image_size]))
        smnoise = filters.gaussian(noise, stdkernel)
        smnoise = smnoise / np.std(smnoise) * stdbckg
        J = I + smnoise

        if scale_intensities_to_one:
            J = utils.map_image_to_intensity_range(
                J, -1, 1, percentiles=norm_percentile)

        Features[:, n+numP1samples+numP2samples] = J.reshape(image_size ** 2)
        Labels[n+numP1samples+numP2samples] = 0

    if save_type == 'text':
        txt_folder = os.path.dirname(out_path)
        np.savetxt(os.path.join(txt_folder, 'features_moving.txt'),
                   Features, fmt='%1.4f')
        np.savetxt(os.path.join(txt_folder, 'labels_moving.txt'),
                   Labels, fmt='%d')
        np.savetxt(os.path.join(
            txt_folder, 'gt_features_moving.txt'), GT, fmt='%d')

    elif save_type == 'pickle':
        np.savez_compressed(out_path, features=Features, labels=Labels, gt=GT)

    elif save_type == 'hdf5':
        with h5py.File(out_path, 'w') as hdf5_file:
            hdf5_file.create_dataset(
                'features', data=Features, dtype=np.float32)
            hdf5_file.create_dataset('labels', data=Labels, dtype=np.uint8)
            hdf5_file.create_dataset('gt', data=GT, dtype=np.uint8)

    else:
        raise ValueError('Unknown save_type: %s' % save_type)


if __name__ == "__main__":
    # import config.system as sys_config
    synth_preproc_folder = "C:\\Users\\cherb\\Documents\\Github\\M-VAE-GAN\\datasets\\synthetic"

    image_size = 216
    effect_size = 100
    num_samples = 20000
    moving_effect = True
    scale_to_one = True

    data = load_and_maybe_generate_data(synth_preproc_folder,
                                        image_size,
                                        effect_size,
                                        num_samples,
                                        moving_effect,
                                        scale_to_one=True,
                                        force_overwrite=False)
    dset = SynthDataset(None, True)
    healthy_dataloader = torch.utils.data.DataLoader(dset, batch_size=64,
                                                     shuffle=True, drop_last=True)
    for batch in healthy_dataloader:
        print(batch)
        break
