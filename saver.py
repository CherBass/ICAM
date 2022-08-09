##################################################
# Author: {Cher Bass}
# Copyright: Copyright {2020}, {ICAM}
# License: {MIT license}
# Credits: {Hsin-Ying Lee}, {2019}, {https://github.com/HsinYingLee/MDMM}
##################################################
import os
import torchvision
from tensorboardX import SummaryWriter
        
class Saver():
    def __init__(self, opts):
        """
        Saver class for saving 2D model, images, and write losses to tensorboard
        :param opts:
        """
        self.model_dir = os.path.join(opts.result_dir, opts.name)
        self.image_dir = os.path.join(self.model_dir, 'images')
        self.display_freq = opts.display_freq

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
        assembled_images, content_images, attr_images = model.assemble_outputs()
        img_filename = '%s/gen_e%04d_i%05d.jpg' % (self.image_dir, ep, total_it)
        torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)
        if not (content_images is None):
            img_filename = '%s/content_e%04d_i%05d.jpg' % (self.image_dir, ep, total_it)
            torchvision.utils.save_image(content_images / 2 + 0.5, img_filename, nrow=1)
        if not (attr_images is None):
            img_filename = '%s/attr_e%04d_i%05d.jpg' % (self.image_dir, ep, total_it)
            torchvision.utils.save_image(attr_images / 2 + 0.5, img_filename, nrow=1)

    def write_model(self, ep, total_it, it, model, epoch=False, model_name='model'):
        if not epoch:
            model.save('%s/%s.pth' % (self.model_dir, model_name), ep, total_it, it)
        else:
            model.save('%s/model_e%04d.pth' % (self.model_dir, ep), ep, total_it, it)
