##################################################
# Author: {Cher Bass}
# Copyright: Copyright {2020}, {ICAM}
# License: {MIT license}
##################################################
import os
from tensorboardX import SummaryWriter
import numpy as np
import nibabel as nib
affine = np.load('affine.npy')


class Saver():
    def __init__(self, opts):
        """
        Saver class for saving 3D model, images, and write losses to tensorboard
        :param opts:
        """
        self.model_dir = os.path.join(opts.result_dir, opts.name)
        self.image_dir = os.path.join(self.model_dir, 'images')
        self.display_freq = opts.display_freq
        self.affine = np.load('affine.npy')
        self.opts = opts

        # make directory
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        # create tensorboard writer
        self.writer = SummaryWriter(log_dir=self.model_dir)

    # write losses and images to tensorboard
    def write_display(self, total_it, model):
        if (total_it + 1) % self.display_freq == 0:
            # write loss
            members = [attr for attr in dir(model) if not callable(getattr(model, attr)) and not attr.startswith("__") and 'loss' in attr]
            for m in members:
                self.writer.add_scalar(m, getattr(model, m), total_it)

    # save result images
    def write_img(self, ep, total_it, model):
        images_a, images_b, images_a1, images_a2, images_a3, images_b1, images_b2, images_b3, images_a_content, \
        images_b_content, images_a_attr, images_b_attr, \
        images_a_clc, images_b_clc, images_a_random, images_b_random, images_a_mask, images_b_mask = model.assemble_outputs_3d()

        img_filename = '%s/real_input0_class0_e%04d_i%05d.nii.gz' % (self.image_dir, ep, total_it)
        image = nib.Nifti1Image(images_a, affine=self.affine)
        nib.save(image, img_filename)

        img_filename = '%s/real_input1_class1_e%04d_i%05d.nii.gz' % (self.image_dir, ep, total_it)
        image = nib.Nifti1Image(images_b, affine=self.affine)
        nib.save(image, img_filename)

        img_filename = '%s/gen_recon_input0_class0_e%04d_i%05d.nii.gz' % (self.image_dir, ep, total_it)
        image = nib.Nifti1Image(images_a1, affine=self.affine)
        nib.save(image, img_filename)

        img_filename = '%s/gen_recon_input1_class1_e%04d_i%05d.nii.gz' % (self.image_dir, ep, total_it)
        image = nib.Nifti1Image(images_b1, affine=self.affine)
        nib.save(image, img_filename)

        img_filename = '%s/gen_clc_input0_class0_e%04d_i%05d.nii.gz' % (self.image_dir, ep, total_it)
        image = nib.Nifti1Image(images_a_clc, affine=self.affine)
        nib.save(image, img_filename)

        img_filename = '%s/gen_clc_input1_class1_e%04d_i%05d.nii.gz' % (self.image_dir, ep, total_it)
        image = nib.Nifti1Image(images_b_clc, affine=self.affine)
        nib.save(image, img_filename)

        img_filename = '%s/gen_random_input0_class1_e%04d_i%05d.nii.gz' % (self.image_dir, ep, total_it)
        image = nib.Nifti1Image(images_a_random, affine=self.affine)
        nib.save(image, img_filename)

        img_filename = '%s/gen_random_input1_class0_e%04d_i%05d.nii.gz' % (self.image_dir, ep, total_it)
        image = nib.Nifti1Image(images_b_random, affine=self.affine)
        nib.save(image, img_filename)

        if not (images_a_content is None):
            img_filename = '%s/content_input0_class0_e%04d_i%05d.nii.gz' % (self.image_dir, ep, total_it)
            image = nib.Nifti1Image(images_a_content, affine=self.affine)
            nib.save(image, img_filename)

            img_filename = '%s/content_input0_class1_e%04d_i%05d.nii.gz' % (self.image_dir, ep, total_it)
            image = nib.Nifti1Image(images_b_content, affine=self.affine)
            nib.save(image, img_filename)

        if not (images_a2 is None):
            img_filename = '%s/gen_fake_input0_class1_e%04d_i%05d.nii.gz' % (self.image_dir, ep, total_it)
            image = nib.Nifti1Image(images_a2, affine=self.affine)
            nib.save(image, img_filename)

            img_filename = '%s/gen_fake_input1_class0_e%04d_i%05d.nii.gz' % (self.image_dir, ep, total_it)
            image = nib.Nifti1Image(images_b2, affine=self.affine)
            nib.save(image, img_filename)

        if not (images_a3 is None):
            img_filename = '%s/gen_diff_pos_input0_class0to1_e%04d_i%05d.nii.gz' % (self.image_dir, ep, total_it)
            image = nib.Nifti1Image(images_a3, affine=self.affine)
            nib.save(image, img_filename)

            img_filename = '%s/gen_diff_pos_input1_class1to0_e%04d_i%05d.nii.gz' % (self.image_dir, ep, total_it)
            image = nib.Nifti1Image(images_b3, affine=self.affine)
            nib.save(image, img_filename)

            img_filename = '%s/gen_diff_neg_input0_class0to1_e%04d_i%05d.nii.gz' % (self.image_dir, ep, total_it)
            image = nib.Nifti1Image(-images_a3, affine=self.affine)
            nib.save(image, img_filename)

            img_filename = '%s/gen_diff_neg_input1_class1to0_e%04d_i%05d.nii.gz' % (self.image_dir, ep, total_it)
            image = nib.Nifti1Image(-images_b3, affine=self.affine)
            nib.save(image, img_filename)

        if not (images_a_mask is None):
            img_filename = '%s/real_input0_mask_e%04d_i%05d.nii.gz' % (self.image_dir, ep, total_it)
            image = nib.Nifti1Image(images_a_mask, affine=self.affine)
            nib.save(image, img_filename)

            img_filename = '%s/real_input1_mask_e%04d_i%05d.nii.gz' % (self.image_dir, ep, total_it)
            image = nib.Nifti1Image(images_b_mask, affine=self.affine)
            nib.save(image, img_filename)

    def write_model(self, ep, total_it, it, model, epoch=False, model_name='model'):
        print('--- save the model @ ep %d ---' % (ep))
        if not epoch:
            model.save('%s/%s.pth' % (self.model_dir, model_name), ep, total_it, it)
        else:
            model.save('%s/model_e%04d.pth' % (self.model_dir, ep), ep, total_it, it)
