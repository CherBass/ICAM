##################################################
# Author: {Cher Bass}
# Copyright: Copyright {2020}, {ICAM}
# License: {MIT license}
##################################################
import torch
from torch.utils.data import Dataset
from synthetic_dataloader import *
import torchvision.transforms as transforms
from skimage.transform import resize
from biobank_dataloader import *
from dhcp_dataloader import *
import torchvision
import SimpleITK as sitk
import random
import numpy as np
from PIL import Image


def line_best_fit(X, Y):
    """
    Line of best fit for variables X, Y
    :param X:
    :param Y:
    :return:
    """

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    return a, b

def whitening(image):
    """Whitening. Normalises image to zero mean and unit variance."""
    image = image.astype(np.float32)
    mean = np.mean(image)
    std = np.std(image)

    if std > 0:
        ret = (image - mean) / std
    else:
        ret = image * 0.
    return ret


def normalise_zero_one(image):
    """Image normalisation. Normalises image to fit [0, 1] range."""

    image = image.astype(np.float32)

    minimum = np.min(image)
    maximum = np.max(image)
    if maximum > minimum:
        ret = (image - minimum) / (maximum - minimum)
    else:
        ret = image * 0.
    return ret


def normalise_negative_one(image):
    """
    Image normalisation. Normalises image to fit [-1, 1] range.
    """
    image = image.astype(np.float32)
    minimum = np.min(image)
    maximum = np.max(image)
    if maximum > minimum:
        ret = (2*(image - minimum) / (maximum-minimum)) - 1
    else:
        ret = image * 0.
    return ret


class NormMinMax(object):
    def __init__(self):
        """
        Normalize image between 0 and 1
        """
        self.int = 1

    def __call__(self, data):
        image = data.astype(np.float32)

        minimum = np.min(image)
        maximum = np.max(image)

        if maximum > minimum:
            ret = (image - minimum) / (maximum - minimum)
        else:
            ret = image * 0.
        return ret


class NormMeanSTD(object):
    def __init__(self, data_mean=None, data_std=None):
        """
        Normalize image with standardization technique.
        :param data_mean: The dataset mean
        :param data_std: The dataset std
        """
        self.data_mean = data_mean
        self.data_std = data_std

    def __call__(self, data):
        if self.data_mean:
            data_norm = (data - self.data_mean) / self.data_std
        else:
            data_norm = data
        return data_norm


class ResizeImage(object):
    def __init__(self, image_size=(128, 160, 128)):
        """    Rescale image- default image size - [128, 160, 128]
        :param image_size:
        """
        self.image_size = image_size

    def __call__(self, data):
        if len(self.image_size) == 2:
            image_resized = resize(data, (self.image_size[0], self.image_size[1]),
                                   anti_aliasing=True)
        else:
            image_resized = resize(data, (self.image_size[0], self.image_size[1], self.image_size[2]),
                                   anti_aliasing=True)

        return image_resized


class RicianNoise(object):
    def __init__(self, noise_level):
        """
        Fourier transformed Gaussian Noise is Rician Noise.
        :param noise_level: The amount of noise to add
        """
        self.noise_level = noise_level

    def add_complex_noise(self, inverse_image, noise_level):
        # Convert the noise from decibels to a linear scale: See: http://www.mogami.com/e/cad/db.html
        noise_level_linear = 10 ** (noise_level / 10)
        # Real component of the noise: The noise "map" should span the entire image, hence the multiplication
        real_noise = np.sqrt(noise_level_linear / 2) * np.random.randn(inverse_image.shape[0],
                                                                       inverse_image.shape[1], inverse_image.shape[2])
        # Imaginary component of the noise: Note the 1j term
        imaginary_noise = np.sqrt(noise_level_linear / 2) * 1j * np.random.randn(inverse_image.shape[0],
                                                                                 inverse_image.shape[1], inverse_image.shape[2])
        noisy_inverse_image = inverse_image + real_noise + imaginary_noise
        return noisy_inverse_image

    def __call__(self, image):
        prob = random.uniform(0, 1)
        if prob > 0.5:
            if len(self.noise_level) == 2:
                noise_level = np.random.randint(self.noise_level[0], self.noise_level[1])
                noise_level = noise_level
            else:
                noise_level = self.noise_level[0]
            # Fourier transform the input image
            inverse_image = np.fft.fftn(image)
            # Add complex noise to the image in k-space
            inverse_image_noisy = self.add_complex_noise(inverse_image, noise_level)
            # Reverse Fourier transform the image back into real space
            complex_image_noisy = np.fft.ifftn(inverse_image_noisy)
            # Calculate the magnitude of the image to get something entirely real
            magnitude_image_noisy = np.sqrt(np.real(complex_image_noisy) ** 2 + np.imag(complex_image_noisy) ** 2)
        else:
            magnitude_image_noisy = image
        return magnitude_image_noisy


class ElasticDeformationsBspline(object):
    def __init__(self, num_controlpoints=5, sigma=1):
        """
        Elastic deformations class
        :param num_controlpoints:
        :param sigma:
        """
        self.num_controlpoints = num_controlpoints
        self.sigma = sigma

    def create_elastic_deformation(self, image, num_controlpoints, sigma):
        """
        We need to parameterise our b-spline transform
        The transform will depend on such variables as image size and sigma
        Sigma modulates the strength of the transformation
        The number of control points controls the granularity of our transform
        """
        # Create an instance of a SimpleITK image of the same size as our image
        itkimg = sitk.GetImageFromArray(np.zeros(image.shape))
        # This parameter is just a list with the number of control points per image dimensions
        trans_from_domain_mesh_size = [num_controlpoints] * itkimg.GetDimension()
        # We initialise the transform here: Passing the image size and the control point specifications
        bspline_transformation = sitk.BSplineTransformInitializer(itkimg, trans_from_domain_mesh_size)
        # Isolate the transform parameters: They will be all zero at this stage
        params = np.asarray(bspline_transformation.GetParameters(), dtype=float)
        # Let's initialise the transform by randomly initialising each parameter according to sigma
        params = params + np.random.randn(params.shape[0]) * sigma
        bspline_transformation.SetParameters(tuple(params))
        return bspline_transformation

    def __call__(self, image):
        prob = random.uniform(0, 1)
        if prob > 0.5:
            if len(self.num_controlpoints) == 2:
                num_controlpoints = np.random.randint(self.num_controlpoints[0], self.num_controlpoints[1])
                num_controlpoints = num_controlpoints
            else:
                num_controlpoints = self.num_controlpoints[0]
            if len(self.sigma) == 2:
                sigma = np.random.uniform(self.sigma[0], self.sigma[1])
                sigma = sigma
            else:
                sigma = self.sigma[0]
            # We need to choose an interpolation method for our transformed image, let's just go with b-spline
            resampler = sitk.ResampleImageFilter()
            resampler.SetInterpolator(sitk.sitkBSpline)
            # Let's convert our image to an sitk image
            sitk_image = sitk.GetImageFromArray(image)
            # sitk_grid = self.create_grid(image)
            # Specify the image to be transformed: This is the reference image
            resampler.SetReferenceImage(sitk_image)
            resampler.SetDefaultPixelValue(0)
            # Initialise the transform
            bspline_transform = self.create_elastic_deformation(image, num_controlpoints, sigma)
            # Set the transform in the initialiser
            resampler.SetTransform(bspline_transform)
            # Carry out the resampling according to the transform and the resampling method
            out_img_sitk = resampler.Execute(sitk_image)
            # out_grid_sitk = resampler.Execute(sitk_grid)
            # Convert the image back into a python array
            out_img = sitk.GetArrayFromImage(out_img_sitk)
            out_img = out_img.reshape(image.shape)
        else:
            out_img = image
        return out_img


class GenHelper(Dataset):
    def __init__(self, mother, length, mapping):
        """
        Class to help with splitting dataloaders into separate datasets
        """
        # here is a mapping from this index to the mother ds index
        self.mapping = mapping
        self.length = length
        self.mother = mother

    def __getitem__(self, index):
        return self.mother[self.mapping[index]]

    def __len__(self):
        return self.length


def train_val_test_split(ds, val_split=0.1, test_split=0.1, random_seed=None):
    '''
    This is a pytorch generic function that takes a data.Dataset object and splits it to train, validation, and test.
    :param ds: data
    :param split_fold: train val split
    :param random_seed: seed
    :return: train, val, test datasets
    '''
    if random_seed != None:
        np.random.seed(random_seed)

    dslen = len(ds)
    indices = list(range(dslen))
    val_size = int(dslen * val_split)
    test_size = int(dslen * test_split)
    train_size = int(dslen-val_size-test_size)
    np.random.shuffle(indices)
    train_mapping = indices[:train_size]
    val_mapping = indices[train_size:train_size+val_size]
    test_mapping = indices[train_size+val_size:train_size+val_size+test_size]
    train = GenHelper(ds, train_size, train_mapping)
    val = GenHelper(ds, val_size, val_mapping)
    test = GenHelper(ds, test_size, test_mapping)

    return train, val, test


def train_val_test_split_dhcp(ds, val_split=0.1, test_split=0.1, random_seed=None):
    '''
    This is a pytorch generic function that takes a data.Dataset object and splits it to train, validation, and test - used only for the 2D dHCP dataset so train/validation and test sets don't share the same subjects.
    :param ds: data
    :param split_fold: train val split
    :param random_seed: seed
    :return: train, val, test datasets
    '''
    if random_seed != None:
        np.random.seed(random_seed)

    dslen = len(ds)
    subj_count = dslen / 10 # 10 slices for each patient
    
    indices_subs = list(range(0,dslen,10)) # 10 slices per subject
    np.random.shuffle(indices_subs)
    
    list_shuff_subs = []
    np.random.shuffle(indices_subs)
    for ind in indices_subs:
        i = 0
        for value in range(10):
            list_shuff_subs.append(ind+i)
            i += 1 
    
    #size of sets
    val_size = int(subj_count * val_split) # will be in subjects (*10 to get slices)
    test_size = int(subj_count * test_split)
    train_size = int(subj_count - val_size - test_size)
    
    train_size_slices = train_size * 10
    val_size_slices = val_size * 10
    test_size_slices = test_size * 10
    
    train_mapping = list_shuff_subs[:train_size_slices]
    val_mapping = list_shuff_subs[train_size_slices : train_size_slices + val_size_slices]
    test_mapping = list_shuff_subs[train_size_slices + val_size_slices : train_size_slices + val_size_slices + test_size_slices]
    
    train = GenHelper(ds, train_size_slices, train_mapping)
    val = GenHelper(ds, val_size_slices, val_mapping)
    test = GenHelper(ds, test_size_slices, test_mapping)

    return train, val, test


def train_valid_split(ds, split_fold=0.1, random_seed=None):
    """
    This is a pytorch generic function that takes a data.Dataset object and splits it to train, validation.
    :param ds: data
    :param split_fold: train val split
    :param random_seed: seed
    :return: train, val datasets
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    dslen = len(ds)
    indices = list(range(dslen))
    valid_size = int(dslen * split_fold)
    np.random.shuffle(indices)
    train_mapping = indices[valid_size:]
    valid_mapping = indices[:valid_size]
    train = GenHelper(ds, dslen - valid_size, train_mapping)
    valid = GenHelper(ds, valid_size, valid_mapping)

    return train, valid


def train_valid_split_dhcp(ds, split_fold=0.1, random_seed=None):
    """
    This is a pytorch generic function that takes a data.Dataset object and splits it to train, validation - - used only for the 2D dHCP dataset so train/validation and test sets don't share the same subjects.
    :param ds: data
    :param split_fold: train val split
    :param random_seed: seed
    :return: train, val datasets
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    dslen = len(ds)
    subj_count = dslen / 10 # 10 slices for each patient
    
    indices_subs = list(range(0,dslen,10)) 
    np.random.shuffle(indices_subs)
    
    list_shuff_subs = []
    np.random.shuffle(indices_subs)
    for ind in indices_subs:
        i = 0
        for value in range(10):
            list_shuff_subs.append(ind+i)
            i += 1 
    
    
    list_shuff_subs = []
    np.random.shuffle(indices_subs)
    
    for ind in indices_subs:
        i = 0
        for value in range(10):
            list_shuff_subs.append(ind+i)
            i += 1 
    
    valid_size = int(subj_count * split_fold) # will be in subjects (*10 to get slices)
    val_size_slices = valid_size * 10
    
    train_mapping = list_shuff_subs[val_size_slices:]
    valid_mapping = list_shuff_subs[:val_size_slices]
    
    train = GenHelper(ds, dslen - val_size_slices, train_mapping)
    valid = GenHelper(ds, val_size_slices, valid_mapping)

    return train, valid


# ---------------------------------------------- dataloaders ------------------------------------------------------


def init_synth_dataloader(opt, anomaly, mode='train', batch_size=2):
    """
    Initialize SynthDataset
    :param opt: options
    :param anomaly: whether squares or no squares
    :param mode: train, val or test
    :param batch_size: batch size
    :return: dataloader
    """
    dataset = SynthDataset(opt=opt, anomaly=anomaly,
                           mode=mode,
                           transform=transforms.Compose([
                               torch.tensor,]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, drop_last=True)

    return dataloader


def init_synth_dataloader_crossval(opt, anomaly, mode='train', batch_size=2):
    """
    Initialize SynthDataset
    :param opt: options
    :param anomaly: whether squares or no squares
    :param mode: train, val or test
    :return: dataset
    """
    dataset = SynthDataset(opt=opt, anomaly=anomaly,
                           mode=mode,
                           transform=transforms.Compose([
                               torch.tensor,]))

    if mode == 'test':
    	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, drop_last=True)
    	return dataloader 
    
    
    elif mode == 'train': 
        return dataset 


def init_biobank_age_dataloader(opt, shuffle_test=False):
    """
    Initialize both datasets and dataloaders
    image_size = [128, 160, 128]

    :param opt: options
    :param shuffle_test: whether to shuffle test data
    :return: dataloader
    """
    if (not opt.aug_rician_noise == None) or (not opt.aug_bspline_deformation == None) \
            or (not opt.resize_image == None):
        transforms = []
    else:
        transforms = None

    if opt.resize_image:
        transforms.append(ResizeImage(image_size=opt.resize_size))

    if opt.aug_rician_noise:
        transforms.append(RicianNoise(noise_level=opt.aug_rician_noise))

    if opt.aug_bspline_deformation:
        transforms.append(ElasticDeformationsBspline(num_controlpoints=opt.aug_bspline_deformation[0],
                                                     sigma=opt.aug_bspline_deformation[1]))

    if opt.aug_rician_noise or opt.aug_bspline_deformation or opt.resize_image:
        transforms = torchvision.transforms.Compose(transforms)

    healthy_train = BiobankRegAgeDataset(image_path=opt.dataroot+'_data',
                                         label_path=opt.label_path,
                                         class_bins=opt.age_range_0,
                                         class_label=0,
                                         get_id=opt.get_id,
                                         transform=transforms)

    anomaly_train = BiobankRegAgeDataset(image_path=opt.dataroot+'_data',
                                         label_path=opt.label_path,
                                         class_bins=opt.age_range_1,
                                         class_label=1,
                                         get_id=opt.get_id,
                                         transform=transforms)

    healthy_dataloader_train, healthy_dataloader_val, healthy_dataloader_test \
        = train_val_test_split(healthy_train, val_split=0.05, test_split=0.05, random_seed=opt.random_seed)
    anomaly_dataloader_train, anomaly_dataloader_val, anomaly_dataloader_test \
        = train_val_test_split(anomaly_train, val_split=0.05, test_split=0.05, random_seed=opt.random_seed)

    print('Train data length: ', len(healthy_dataloader_train), 'Val data length: ',
          len(healthy_dataloader_val), 'Test data length: ', len(healthy_dataloader_test))
    print('Train data length: ', len(anomaly_dataloader_train), 'Val data length: ',
          len(anomaly_dataloader_val), 'Test data length: ', len(anomaly_dataloader_test))

    healthy_dataloader_train = torch.utils.data.DataLoader(healthy_dataloader_train, batch_size=opt.batch_size//2,
                                                           shuffle=True)
    anomaly_dataloader_train = torch.utils.data.DataLoader(anomaly_dataloader_train, batch_size=opt.batch_size//2,
                                                           shuffle=True)

    healthy_dataloader_val = torch.utils.data.DataLoader(healthy_dataloader_val, batch_size=opt.batch_size//2,
                                                         shuffle=True)
    anomaly_dataloader_val = torch.utils.data.DataLoader(anomaly_dataloader_val, batch_size=opt.batch_size//2,
                                                         shuffle=True)
    healthy_dataloader_test = torch.utils.data.DataLoader(healthy_dataloader_test, batch_size=opt.batch_size//2,
                                                         shuffle=shuffle_test)
    anomaly_dataloader_test = torch.utils.data.DataLoader(anomaly_dataloader_test, batch_size=opt.batch_size//2,
                                                         shuffle=shuffle_test)

    return healthy_dataloader_train, healthy_dataloader_val, healthy_dataloader_test, \
           anomaly_dataloader_train, anomaly_dataloader_val, anomaly_dataloader_test


def init_biobank_age_dataloader_crossval(opt, shuffle_test=False):
    """
    Initialize both datasets and dataloaders
    image_size = [128, 160, 128]

    :param opt: options
    :param shuffle_test: whether to shuffle test data
    :return: dataloader
    """
    if (not opt.aug_rician_noise == None) or (not opt.aug_bspline_deformation == None) \
            or (not opt.resize_image == None):
        transforms = []
    else:
        transforms = None

    if opt.resize_image:
        transforms.append(ResizeImage(image_size=opt.resize_size))

    if opt.aug_rician_noise:
        transforms.append(RicianNoise(noise_level=opt.aug_rician_noise))

    if opt.aug_bspline_deformation:
        transforms.append(ElasticDeformationsBspline(num_controlpoints=opt.aug_bspline_deformation[0],
                                                     sigma=opt.aug_bspline_deformation[1]))

    if opt.aug_rician_noise or opt.aug_bspline_deformation or opt.resize_image:
        transforms = torchvision.transforms.Compose(transforms)

    healthy_train = BiobankRegAgeDataset(image_path=opt.dataroot+'_data',
                                         label_path=opt.label_path,
                                         class_bins=opt.age_range_0,
                                         class_label=0,
                                         get_id=opt.get_id,
                                         transform=transforms)

    anomaly_train = BiobankRegAgeDataset(image_path=opt.dataroot+'_data',
                                         label_path=opt.label_path,
                                         class_bins=opt.age_range_1,
                                         class_label=1,
                                         get_id=opt.get_id,
                                         transform=transforms)

    healthy_dataset_train, healthy_dataset_test = train_valid_split(healthy_train, split_fold=0.1,
                                                                         random_seed=opt.random_seed)  #90/10 for train/test
    anomaly_dataset_train, anomaly_dataset_test = train_valid_split(anomaly_train, split_fold=0.1,
                                                                         random_seed=opt.random_seed) #90/10 for train/test
    
    print('Full Train healthy data length in fold: ', len(healthy_dataset_train), 'Test data hold-out length: ',len(healthy_dataset_test))
    print('Full Train anomaly data length in fold: ', len(anomaly_dataset_train), 'Test data hold-out length: ',len(anomaly_dataset_test))


    healthy_dataloader_test = torch.utils.data.DataLoader(healthy_dataset_test, batch_size=opt.batch_size//2,
                                                         shuffle=shuffle_test)
    anomaly_dataloader_test = torch.utils.data.DataLoader(anomaly_dataset_test, batch_size=opt.batch_size//2,
                                                         shuffle=shuffle_test)

    return healthy_dataset_train, healthy_dataloader_test, anomaly_dataset_train, anomaly_dataloader_test


def init_dhcp_dataloader_2d_crossval(opt, shuffle_test=False):
    '''
    Initialize both datasets and dataloaders
    image_size = [128, 160]
    '''
    if (not opt.aug_rician_noise == None) or (not opt.aug_bspline_deformation == None) or (not opt.resize_image == None):
        transforms = []
    else:
        transforms = None

    if opt.resize_image:
        transforms.append(ResizeImage(image_size=opt.resize_size))
        

    if opt.aug_rician_noise:
        transforms.append(RicianNoise(noise_level=opt.aug_rician_noise))

    if opt.aug_bspline_deformation:
        transforms.append(ElasticDeformationsBspline(num_controlpoints=opt.aug_bspline_deformation[0], sigma=opt.aug_bspline_deformation[1]))

    if opt.aug_rician_noise or opt.aug_bspline_deformation or opt.resize_image:
        transforms = torchvision.transforms.Compose(transforms)

    healthy_train = DHCP_2D(image_path=opt.dataroot,
                         label_path=opt.label_path,
                         num_classes=2,
                         task='regression',
                         class_label=0,
                         transform=transforms)

    anomaly_train = DHCP_2D(image_path=opt.dataroot,
                         label_path=opt.label_path,
                         num_classes=2,
                         task='regression',
                         class_label=1,
                         transform=transforms)

    healthy_dataset_train, healthy_dataset_test = train_valid_split_dhcp(healthy_train, split_fold=0.1,
                                                                         random_seed=opt.random_seed)  
    anomaly_dataset_train, anomaly_dataset_test = train_valid_split_dhcp(anomaly_train, split_fold=0.1,
                                                                         random_seed=opt.random_seed) 


    print('Full Train healthy data length in fold: ', len(healthy_dataset_train), 'Test data hold-out length: ',len(healthy_dataset_test))
    print('Full Train anomaly data length in fold: ', len(anomaly_dataset_train), 'Test data hold-out length: ',len(anomaly_dataset_test))
    
    healthy_dataloader_test = torch.utils.data.DataLoader(healthy_dataset_test, batch_size=opt.batch_size//2,
                                                         shuffle=shuffle_test)
    anomaly_dataloader_test = torch.utils.data.DataLoader(anomaly_dataset_test, batch_size=opt.batch_size//2,
                                                         shuffle=shuffle_test)

    return healthy_dataset_train, healthy_dataloader_test, anomaly_dataset_train, anomaly_dataloader_test


def init_dhcp_dataloader_2d(opt, shuffle_test=False):
    '''
    Initialize both datasets and dataloaders
    image_size = [128, 160]
    '''
    if (not opt.aug_rician_noise == None) or (not opt.aug_bspline_deformation == None) or (not opt.resize_image == None):
        transforms = []
    else:
        transforms = None

    if opt.resize_image:
        transforms.append(ResizeImage(image_size=opt.resize_size))

    if opt.aug_rician_noise:
        transforms.append(RicianNoise(noise_level=opt.aug_rician_noise))

    if opt.aug_bspline_deformation:
        transforms.append(ElasticDeformationsBspline(num_controlpoints=opt.aug_bspline_deformation[0], sigma=opt.aug_bspline_deformation[1]))

    if opt.aug_rician_noise or opt.aug_bspline_deformation or opt.resize_image:
        transforms = torchvision.transforms.Compose(transforms)

    healthy_train = DHCP_2D(image_path=opt.dataroot,
                         label_path=opt.label_path,
                         num_classes=2,
                         task='regression',
                         class_label=0,
                         transform=transforms)

    anomaly_train = DHCP_2D(image_path=opt.dataroot,
                         label_path=opt.label_path,
                         num_classes=2,
                         task='regression',
                         class_label=1,
                         transform=transforms)

    healthy_dataloader_train, healthy_dataloader_val, healthy_dataloader_test = train_val_test_split_dhcp(healthy_train, val_split=0.1, test_split=0.1,
                                                                         random_seed=opt.random_seed)
    anomaly_dataloader_train, anomaly_dataloader_val, anomaly_dataloader_test = train_val_test_split_dhcp(anomaly_train, val_split=0.1, test_split=0.1,
                                                                         random_seed=opt.random_seed)


    print('Train healthy data length: ', len(healthy_dataloader_train), 'Val data length: ',len(healthy_dataloader_val), 'Test data length: ', len(healthy_dataloader_test))
    print('Train anomaly data length: ', len(anomaly_dataloader_train), 'Val data length: ',len(anomaly_dataloader_val), 'Test data length: ', len(anomaly_dataloader_test))

    healthy_dataloader_train = torch.utils.data.DataLoader(healthy_dataloader_train, batch_size=opt.batch_size//2,
                                                           shuffle=True)
    anomaly_dataloader_train = torch.utils.data.DataLoader(anomaly_dataloader_train, batch_size=opt.batch_size//2,
                                                           shuffle=True)

    healthy_dataloader_val = torch.utils.data.DataLoader(healthy_dataloader_val, batch_size=opt.batch_size//2,
                                                         shuffle=True)
    anomaly_dataloader_val = torch.utils.data.DataLoader(anomaly_dataloader_val, batch_size=opt.batch_size//2,
                                                         shuffle=True)
    healthy_dataloader_test = torch.utils.data.DataLoader(healthy_dataloader_test, batch_size=opt.batch_size//2,
                                                         shuffle=shuffle_test)
    anomaly_dataloader_test = torch.utils.data.DataLoader(anomaly_dataloader_test, batch_size=opt.batch_size//2,
                                                         shuffle=shuffle_test)

    return healthy_dataloader_train, healthy_dataloader_val, healthy_dataloader_test, anomaly_dataloader_train, anomaly_dataloader_val, anomaly_dataloader_test


