##################################################
# Author: {Cher Bass}
# Copyright: Copyright {2020}, {ICAM}
# License: {MIT license}
# Credits: {Hsin-Ying Lee}, {2019}, {https://github.com/HsinYingLee/MDMM}
##################################################
import argparse


class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # data loader related
        self.parser.add_argument('--dataroot', type=str, default='./datasets' ,
                                 help='path to data')
        self.parser.add_argument('--data_type', type=str, default= 'syn2d' , choices=['syn2d', 'biobank_age', 'dhcp_2d' , 'syn2d_crossval', 'biobank_age_crossval', 'dhcp_2d_crossval'] ,
                                 help='data to load' 
                                      'options: syn2d [128, 128]'
                                      'biobank_age [128, 160, 128]'
                                      'dhcp [128, 128]' 
                                      'add more dataloaders here')
        self.parser.add_argument('--cross_validation', type=bool, default=False, help='wheter to use cross validation (5 kfolds to split data)')
        self.parser.add_argument('--data_dim', type=str, default='2d', choices=['2d', '3d'],
                                 help='whether to load 2d or 3d networks. Options: 2d, 3d')

        # biobank related
        self.parser.add_argument('--label_path', type=str, default= '/data/biobank/biobank_labels_filtered.pkl',
                                 help='path of data')
        self.parser.add_argument('--aug_rician_noise', type=int, default=0,
                                 help='whether to use rician noise augmentation'
                                      'options: 0, ([0, 10])')
        self.parser.add_argument('--aug_bspline_deformation', type=float, default=0,
                                 help='whether to use bspline_deformation'
                                      'options: 0, ([5],[0, 2])')
        self.parser.add_argument('--get_id', type=bool, default=False, help='get subject id during testing')

        # ouptput related
        self.parser.add_argument('--result_dir', type=str, default='./results',
                                 help='path for saving result images and models')
        self.parser.add_argument('--train_print_it', type=int, default=100, help='train print (every x iterations)')
        self.parser.add_argument('--model_save_freq', type=int, default=10, help='freq (epoch) of saving models')
        self.parser.add_argument('--display_freq', type=int, default=1000, help='freq (iteration) of display')

        # network related - experiment params
        self.parser.add_argument('--tch', type=int, default=16, help='# number of (starting) channels')
        self.parser.add_argument('--input_dim', type=int, default=1, help='# of input channels for each domain')
        self.parser.add_argument('--num_domains', type=int, default=2, help='# number of classes')
        self.parser.add_argument('--D_content_dis_cls_all1', type=float, default=0.5, choices=[0, 1, 0.5],
                                 help='whether content encoder tries to classify all classes the same- '
                                                                                           'options: 0, 1, 0.5')
        # optimizer related
        self.parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
        self.parser.add_argument('--lr_dcontent', type=float, default=0.00004,
                                 help='learning rate for content discriminator')
        self.parser.add_argument('--opt_weight_decay', type=float, default=0.0001, help='weight decay')
        self.parser.add_argument('--betas', type=float, default=(0.5, 0.999), help='betas 0.9, 0.999')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='type of learn rate decay. '
                                                                                 'Options: step, lambda')
        self.parser.add_argument('--n_ep_decay', type=int, default=-1, help='epoch start decay learning rate, '
                                                                            'set -1 if no decay')

        # training related
        self.parser.add_argument('--batch_size', type=int, default=2, help='train batch size')
        self.parser.add_argument('--val_batch_size', type=int, default=2, help='val batch size')
        self.parser.add_argument('--n_ep', type=int, default=300, help='number of epochs')
        self.parser.add_argument('--d_iter', type=int, default=3,
                                 help='# of iterations for updating content discriminator ')
        self.parser.add_argument('--dis_scale', type=int, default=3,
                                 help='scale of discriminator')
        self.parser.add_argument('--dis_norm', type=str, default='None',
                                 help='normalization layer in discriminator [None, Instance]')
        self.parser.add_argument('--dis_spectral_norm', action='store_true',
                                 help='use spectral normalization in discriminator')
        self.parser.add_argument('--lambda_rec', type=float, default=100)
        self.parser.add_argument('--lambda_rec_cc', type=float, default=100)
        self.parser.add_argument('--lambda_l2_rec', type=float, default=100)
        self.parser.add_argument('--lambda_l2_rec_cc', type=float, default=100)
        self.parser.add_argument('--lambda_cls_D', type=float, default=1)
        self.parser.add_argument('--lambda_cls_E', type=float, default=10)
        self.parser.add_argument('--lambda_cls_G', type=float, default=5)
        self.parser.add_argument('--lambda_D_gan', type=float, default=1)
        self.parser.add_argument('--lambda_E_content_cls', type=float, default=1)
        self.parser.add_argument('--lambda_D_content_cls', type=float, default=1)
        self.parser.add_argument('--lambda_G_gan', type=float, default=1)
        self.parser.add_argument('--lambda_diff_M_reg', type=float, default=10)
        self.parser.add_argument('--lambda_latent_l1', type=float, default=1)
        self.parser.add_argument('--lambda_kl_zc', type=float, default=0.01)
        self.parser.add_argument('--lambda_kl_za', type=float, default=0.01)
        self.parser.add_argument('--cross_corr', type=bool, default=True, help='whether to check cross correlation '
                                                                                'only suitable for datasets with masks')
        self.parser.add_argument('--loss_latent_l1_random', type=bool, default=True, help='latent regression loss')
        self.parser.add_argument('--loss_diff_M', type=bool, default=True, help='feature attribution map loss')
        self.parser.add_argument('--rejection_sampling', type=bool, default=True, help='whether to predict class from '
                                                                                       'random z latent, '
                                                                                       'and feed to generator z '
                                                                                       'from correct class')
        self.parser.add_argument('--regression', type=bool, default=False, help='whether to add another regression loss'
                                                                                ' in the attribute encoder')
        self.parser.add_argument('--gpu', type=bool, default=True, help='whether to use gpu')
        self.parser.add_argument('--device', type=int, default=0, help='which device num to use: 0,1,2')

    def parse(self):
        opt = self.parser.parse_args()
        args = vars(opt)
        print('\n--- load train options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return opt

