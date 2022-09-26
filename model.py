##################################################
# Author: {Cher Bass}
# Copyright: Copyright {2020}, {ICAM}
# License: {MIT license}
# Credits: {Hsin-Ying Lee}, {2019}, {https://github.com/HsinYingLee/MDMM}
##################################################
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from dataloader_utils import *
import torch.nn.functional as F


class ICAM(nn.Module):
    def __init__(self, opts):
        """
        ICAM framework - initialize all networks, optimizers, and losses
        :param opts: parameters
        """
        super(ICAM, self).__init__()
        self.opts = opts
        self.nz = opts.nz

        global networks
        if opts.data_dim == '3d':
            import networks_3d as networks
        else:
            import networks as networks

        self.dis1 = networks.NetDis(opts)
        self.dis2 = networks.NetDis(opts)
        self.enc_c = networks.NetEc(opts)
        self.gen = networks.NetGen(opts)
        self.enc_a = networks.NetEa(opts)
        self.disContent = networks.NetDisContent(opts)

        self.dis1_opt = torch.optim.Adam(self.dis1.parameters(), lr=opts.lr, betas=(opts.betas[0], opts.betas[1]), weight_decay=opts.opt_weight_decay)
        self.dis2_opt = torch.optim.Adam(self.dis2.parameters(), lr=opts.lr, betas=(opts.betas[0], opts.betas[1]), weight_decay=opts.opt_weight_decay)
        self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), lr=opts.lr, betas=(opts.betas[0], opts.betas[1]), weight_decay=opts.opt_weight_decay)
        self.enc_a_opt = torch.optim.Adam(self.enc_a.parameters(), lr=opts.lr, betas=(opts.betas[0], opts.betas[1]), weight_decay=opts.opt_weight_decay)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=opts.lr, betas=(opts.betas[0], opts.betas[1]), weight_decay=opts.opt_weight_decay)
        self.disContent_opt = torch.optim.Adam(self.disContent.parameters(), lr=opts.lr_dcontent, betas=(opts.betas[0], opts.betas[1]), weight_decay=opts.opt_weight_decay)

        # loss functions
        self.cls_loss = nn.BCEWithLogitsLoss()
        if self.opts.lambda_l2_rec > 0:
            self.l2_loss = nn.MSELoss()
        if self.opts.regression:
            self.reg_loss = nn.SmoothL1Loss()

    def initialize(self):
        """
        Initialize network weights
        :return:
        """
        self.dis1.apply(networks.gaussian_weights_init)
        self.dis2.apply(networks.gaussian_weights_init)
        self.disContent.apply(networks.gaussian_weights_init)
        self.gen.apply(networks.gaussian_weights_init)
        self.enc_c.apply(networks.gaussian_weights_init)
        self.enc_a.apply(networks.gaussian_weights_init)

    def set_scheduler(self, opts, last_ep=0):
        """
        Scheduler for learning rates
        :param opts:
        :param last_ep:
        :return:
        """
        self.dis1_sch = networks.get_scheduler(self.dis1_opt, opts, last_ep)
        self.dis2_sch = networks.get_scheduler(self.dis2_opt, opts, last_ep)
        self.disContent_sch = networks.get_scheduler(self.disContent_opt, opts, last_ep)
        self.enc_c_sch = networks.get_scheduler(self.enc_c_opt, opts, last_ep)
        self.enc_a_sch = networks.get_scheduler(self.enc_a_opt, opts, last_ep)
        self.gen_sch = networks.get_scheduler(self.gen_opt, opts, last_ep)

    def setgpu(self, device):
        """
        Setup gpu device
        :param device: number
        :return:
        """
        self.device = device
        self.dis1.to(self.device)
        self.dis2.to(self.device)
        self.enc_c.to(self.device)
        self.enc_a.to(self.device)
        self.gen.to(self.device)
        self.disContent.to(self.device)

    def _get_z_random(self, batchSize, nz, random_type='gauss'):
        """
        Sample random z vector
        :param batchSize: batch size
        :param nz: vector size
        :param random_type:
        :return:
        """
        z = torch.randn(batchSize, nz).to(self.device)
        return z

    def classification_scores(self, image, c_org):
        """
        Classification score for predictions
        :param image: input image
        :param c_org: class label (one hot vector)
        :return:
        """
        _, _, E_pred_cls, _ = self.enc_a.forward(image)
        _, y_pred = torch.max(E_pred_cls, 1)
        _, y_true = torch.max(c_org, 1)
        y_true = y_true.data.cpu().numpy()
        y_pred = y_pred.data.cpu().numpy()
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        return accuracy, f1, precision, recall

    def regression(self, image, c_org):
        """
        Regression score for predictions
        :param image: input image
        :param c_org: regression label
        :return:
        """
        _, _, _, E_pred_reg = self.enc_a.forward(image)
        mse = F.mse_loss(E_pred_reg.detach(), c_org)
        mae = F.l1_loss(E_pred_reg.detach(), c_org)
        mse = mse.cpu().numpy().astype(float)
        mae = mae.cpu().numpy().astype(float)
        pred = E_pred_reg.squeeze(1).detach().cpu().numpy().astype(float)
        return mse, mae, pred

    def cross_correlation(self, image, mask, c_org, brain_mask_a=None, brain_mask_b=None):
        """
        Cross correlation code between generated feature attribution map (FA) and mask image
        :param image: FA map - for image a and b
        :param mask: mask
        :param c_org: label
        :param brain_mask_a: brain mask for image a if available
        :param brain_mask_b: brain mask for image b if available
        :return:
        """
        half_size = image.size(0) // 2
        real_A = image[0:half_size]
        real_B = image[half_size:]
        c_org_A = c_org[0:half_size]
        c_org_B = c_org[half_size:]

        z_content = self.enc_c.forward(image)
        z_content_a, z_content_b = torch.split(z_content, half_size, dim=0)
        mu, logvar, E_pred_cls, _ = self.enc_a.forward(image)
        std = logvar.mul(0.5).exp_()
        eps = self._get_z_random(std.size(0), std.size(1), 'gauss')
        z_attr = eps.mul(std).add_(mu)
        z_attr_a, z_attr_b = torch.split(z_attr, half_size, dim=0)

        output_fakeA = self.gen.forward(x=z_content_b, z=z_attr_a, c=c_org_A)
        output_fakeB = self.gen.forward(x=z_content_a, z=z_attr_b, c=c_org_B)

        diff_fake_A_encoded = (output_fakeA - real_B)
        diff_fake_B_encoded = (output_fakeB - real_A)

        if not brain_mask_a is None:
            diff_fake_A_encoded = diff_fake_A_encoded * brain_mask_a * brain_mask_b
            diff_fake_B_encoded = diff_fake_B_encoded * brain_mask_a * brain_mask_b

        diff_fake_A_encoded = diff_fake_A_encoded.detach().cpu().numpy().flatten()
        diff_fake_B_encoded = diff_fake_B_encoded.detach().cpu().numpy().flatten()
        mask = mask.detach().cpu().numpy().flatten()

        diff_fake_A_encoded = abs(diff_fake_A_encoded)
        diff_fake_B_encoded = abs(diff_fake_B_encoded)
        mask = abs(mask)

        diff_fake_A_encoded = (diff_fake_A_encoded - np.mean(diff_fake_A_encoded)) / (np.std(diff_fake_A_encoded) * len(diff_fake_A_encoded))
        diff_fake_B_encoded = (diff_fake_B_encoded - np.mean(diff_fake_B_encoded)) / (np.std(diff_fake_B_encoded) * len(diff_fake_B_encoded))
        mask = (mask - np.mean(mask)) / (np.std(mask))
        cross_corr_a = np.mean(np.correlate(diff_fake_A_encoded.flatten(), mask.flatten()))
        cross_corr_b = np.mean(np.correlate(diff_fake_B_encoded.flatten(), mask.flatten()))
        return cross_corr_a, cross_corr_b

    def _rejection_sampling(self, half_size):
        """
        rejection sampling - sample an attribute latent space of an appropriate class by testing with e_a classifier
        :param half_size:
        :return:
        """
        if self.opts.rejection_sampling:
            self.z_random_a = torch.zeros((self.z_attr_a.size(0), self.z_attr_a.size(1))).to(self.device)
            self.z_random_b = torch.zeros((self.z_attr_a.size(0), self.z_attr_a.size(1))).to(self.device)
            flag = True
            i = 0
            j = 0
            k = 0
            while flag:
                k = k + 1
                z_random = self._get_z_random(1, self.nz, 'gauss')
                _, _, pred_random, _ = self.enc_a.forward(x=None, z=z_random.detach())
                prob, pred_ind = torch.max(pred_random, 1)
                if (pred_ind == 0) and (i < half_size):
                    self.z_random_a[i] = z_random
                    i = i + 1
                elif (pred_ind == 1) and (j < half_size):
                    self.z_random_b[j] = z_random
                    j = j + 1
                if (j == half_size) and (i == half_size):
                    flag = False
                elif k > int(50 * 2 * half_size):
                    if (i < half_size):
                        self.z_random_a = self._get_z_random(half_size, self.nz, 'gauss')
                    if (j < half_size):
                        self.z_random_b = self._get_z_random(half_size, self.nz, 'gauss')
                    print('Random z not separable')
                    flag = False
        else:
            self.z_random_a = self._get_z_random(half_size, self.nz, 'gauss')
            self.z_random_b = self.z_random_a

    def forward(self):
        """
        Forward function of ICAM with cross-translation
        :return:
        """
        # input images
        if not self.input.size(0) % 2 == 0:
            print("Need to be even QAQ")
            input()
        half_size = self.input.size(0) // 2
        self.real_A = self.input[0:half_size]
        self.real_B = self.input[half_size:]

        if not self.mask is None:
            self.mask_a = self.mask[0:half_size]
            self.mask_b = self.mask[half_size:]

        c_org_A = self.c_org[0:half_size]
        c_org_B = self.c_org[half_size:]
        self.c_org_A = c_org_A
        self.c_org_B = c_org_B
        # get encoded z_c
        self.real_img = torch.cat((self.real_A, self.real_B), 0)

        self.z_content = self.enc_c.forward(self.real_img)
        self.z_content_a, self.z_content_b = torch.split(self.z_content, half_size, dim=0)
        self.mu, self.logvar, self.E_pred_cls, self.E_pred_reg = self.enc_a.forward(self.real_img)
        self.mu_a, self.mu_b = torch.split(self.mu, half_size, 0)
        std = self.logvar.mul(0.5).exp_()
        eps = self._get_z_random(std.size(0), std.size(1), 'gauss')
        self.z_attr = eps.mul(std).add_(self.mu)
        self.z_attr_a, self.z_attr_b = torch.split(self.z_attr, half_size, dim=0)

        # rejection sampling
        self._rejection_sampling(half_size)
        torch.cuda.empty_cache()  # clear cached GPU memory

        # first cross translation
        input_content_forA = torch.cat((self.z_content_b, self.z_content_a, self.z_content_b), 0)
        input_content_forB = torch.cat((self.z_content_a, self.z_content_b, self.z_content_a), 0)
        input_attr_forA = torch.cat((self.z_attr_a, self.z_attr_a, self.z_random_a), 0)
        input_attr_forB = torch.cat((self.z_attr_b, self.z_attr_b, self.z_random_b), 0)

        output_fakeA = self.gen.forward(x=input_content_forA, z=input_attr_forA, c=None)
        output_fakeB = self.gen.forward(x=input_content_forB, z=input_attr_forB, c=None)

        self.fake_A_encoded, self.fake_AA_encoded, self.fake_A_random = torch.split(output_fakeA,
                                                                                    self.z_content_a.size(0), dim=0)
        self.fake_B_encoded, self.fake_BB_encoded, self.fake_B_random = torch.split(output_fakeB,
                                                                                    self.z_content_a.size(0), dim=0)

        # generate difference map
        self.diff_fake_A_encoded = self.fake_A_encoded - self.real_B
        self.diff_fake_B_encoded = self.fake_B_encoded - self.real_A

        self.fake_encoded_img = torch.cat((self.fake_A_encoded, self.fake_B_encoded), 0)
        self.fake_random_img = torch.cat((self.fake_A_random, self.fake_B_random), 0)

        # get reconstructed encoded z_c
        self.fake_recon_img = torch.cat((self.fake_AA_encoded, self.fake_BB_encoded), 0)
        self.z_content_recon = self.enc_c.forward(self.fake_encoded_img)
        self.z_content_recon_b, self.z_content_recon_a = torch.split(self.z_content_recon, half_size, dim=0)

        # get reconstructed encoded z_a
        self.mu_recon, self.logvar_recon, self.E_pred_cls_fake, self.E_pred_reg_fake = self.enc_a.forward(self.fake_encoded_img)
        torch.cuda.empty_cache()  # clear cached GPU memory
        self.mu_a_recon, self.mu_b_recon = torch.split(self.mu_recon, half_size, 0)
        _, _, self.E_pred_cls_recon, self.E_pred_reg_recon = self.enc_a.forward(self.fake_recon_img)
        torch.cuda.empty_cache()  # clear cached GPU memory
        std_recon = self.logvar_recon.mul(0.5).exp_()
        eps_recon = self._get_z_random(std_recon.size(0), std_recon.size(1), 'gauss')
        self.z_attr_recon = eps_recon.mul(std_recon).add_(self.mu_recon)
        self.z_attr_recon_a, self.z_attr_recon_b = torch.split(self.z_attr_recon, half_size, dim=0)

        # second cross translation
        self.fake_A_recon = self.gen.forward(x=self.z_content_recon_a, z=self.z_attr_recon_a, c=None)
        torch.cuda.empty_cache()  # clear cached GPU memory
        self.fake_B_recon = self.gen.forward(x=self.z_content_recon_b, z=self.z_attr_recon_b, c=None)
        torch.cuda.empty_cache()  # clear cached GPU memory

        # latent regression
        if self.opts.loss_latent_l1_random:
            self.mu2, _, self.E_pred_cls_rand, self.E_pred_reg_rand = self.enc_a.forward(self.fake_random_img)
            self.mu2_a, self.mu2_b = torch.split(self.mu2, half_size, 0)

    def update_D_content(self, opts, image, c_org):
        """
        Update content discriminator weights - forward, accumulate losses, backward
        :param opts:
        :param image: input image
        :param c_org: label
        :return:
        """
        self.opts = opts
        self.input = image
        z_content = self.enc_c.forward(image)

        self.disContent_opt.zero_grad()
        pred_cls = self.disContent.forward(z_content.detach(), mode='cls')

        # goal is to classify correctly
        loss_D_content = self.cls_loss(pred_cls, c_org) * self.opts.lambda_D_content_cls
        loss_D_content.backward()
        self.D_content_loss = loss_D_content.item()
        nn.utils.clip_grad_norm_(self.disContent.parameters(), 5)
        self.disContent_opt.step()

    def update_D(self, opts, image, c_org, c_reg, mask=None):
        """
         Update domain discriminator weights - forward, accumulate losses, backward
        :param opts:
        :param image: input image
        :param c_org: class label
        :param c_reg: regression label
        :param mask: mask image
        :return:
        """
        self.opts = opts
        self.input = image
        self.c_org = c_org
        self.c_reg = c_reg
        self.mask = mask

        self.forward()
        # discriminator on fake encoded (i.e. from real attr latent)
        self.dis1_opt.zero_grad()
        self.D1_gan_loss, self.D1_cls_loss = self.backward_D(self.dis1, self.input, self.fake_encoded_img)
        self.dis1_opt.step()

        # discriminator on fake random (i.e. from random attr latent)
        self.dis2_opt.zero_grad()
        self.D2_gan_loss, self.D2_cls_loss = self.backward_D(self.dis2, self.input, self.fake_random_img)
        self.dis2_opt.step()

    def backward_D(self, netD, real, fake):
        """
         Update domain discriminator weights - forward, accumulate losses, backward
        :param netD: discriminator network
        :param real: real input image
        :param fake: fake input image
        :return:
        """
        pred_fake, pred_fake_cls = netD.forward(fake.detach())
        pred_real, pred_real_cls = netD.forward(real)

        loss_D_gan = 0
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = torch.sigmoid(out_a)
            out_real = torch.sigmoid(out_b)
            all0 = torch.zeros_like(out_fake).to(self.device)
            all1 = torch.ones_like(out_real).to(self.device)
            ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
            ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
            loss_D_gan += ad_true_loss + ad_fake_loss

        loss_D_cls = self.cls_loss(pred_real_cls, self.c_org)

        loss_D_gan = self.opts.lambda_D_gan * loss_D_gan
        loss_D_cls = self.opts.lambda_cls_D * loss_D_cls
        loss_D = loss_D_gan + loss_D_cls
        loss_D.backward()
        return loss_D_gan, loss_D_cls

    def update_EG(self, opts):
        """
         Update encoders (e_c, e_a) and generator weights - accumulate losses, backward
        :param opts:
        :return:
        """
        # update G, Ec, Ea - update with real images
        self.opts = opts
        self.enc_c_opt.zero_grad()
        self.enc_a_opt.zero_grad()
        self.gen_opt.zero_grad()
        self.backward_EG()

        # update G - generator loss on fake generated images
        self.backward_G_alone()
        self.enc_c_opt.step()
        self.enc_a_opt.step()
        self.gen_opt.step()

    def backward_EG(self):
        """
        Accumulate all losses on real images for encoders (e_a, e_c) and generator, backward
        :return:
        """

        # self recon
        loss_G = torch.mean(
            torch.abs(self.input - torch.cat((self.fake_AA_encoded, self.fake_BB_encoded), 0))) * self.opts.lambda_rec
        self.l1_self_rec_loss = loss_G.item()

        # l2 recon + cyclic
        if self.opts.lambda_l2_rec > 0:
            l2_self_rec_loss = self.l2_loss(self.input, torch.cat((self.fake_AA_encoded, self.fake_BB_encoded), 0)) * self.opts.lambda_l2_rec
            self.l2_self_rec_loss = l2_self_rec_loss.item()
            loss_G += l2_self_rec_loss
        if self.opts.lambda_l2_rec_cc > 0:
            l2_cc_rec_loss = self.l2_loss(self.input, torch.cat((self.fake_A_recon, self.fake_B_recon), 0)) * self.opts.lambda_l2_rec_cc
            self.l2_cc_rec_loss = l2_cc_rec_loss.item()
            loss_G += l2_cc_rec_loss

        # content loss
        loss_E_content = self.backward_E_content(self.z_content)
        loss_G += loss_E_content
        self.E_content_loss = loss_E_content.item()

        # discriminator loss
        pred_fake, pred_fake_cls = self.dis1.forward(self.fake_encoded_img)
        loss_G_GAN = 0
        for out_a in pred_fake:
            outputs_fake = torch.sigmoid(out_a)
            all_ones = torch.ones_like(outputs_fake).to(self.device)
            loss_G_GAN += nn.functional.binary_cross_entropy(outputs_fake, all_ones)

        loss_G_gan = loss_G_GAN * self.opts.lambda_G_gan
        self.G_gan_loss = loss_G_gan.item()
        loss_G += loss_G_gan

        # classification
        loss_G_cls = self.cls_loss(pred_fake_cls, self.c_org) * self.opts.lambda_cls_G
        self.G_gan_cls_loss = loss_G_cls.item()
        loss_G += loss_G_cls

        #cross-cycle recon
        loss_G_L1_cc = torch.mean(
            torch.abs(self.input - torch.cat((self.fake_A_recon, self.fake_B_recon), 0))) * self.opts.lambda_rec_cc
        loss_G += loss_G_L1_cc
        self.l1_cc_rec_loss = loss_G_L1_cc.item()

        # KL loss - z_c
        loss_kl_zc = self._l2_regularize(self.z_content) * self.opts.lambda_kl_zc

        # KL loss - z_a
        kl_element = self.mu.pow(2).add_(self.logvar.exp()).mul_(-1).add_(1).add_(self.logvar)
        loss_kl_za = torch.sum(kl_element).mul_(-0.5) * self.opts.lambda_kl_za
        self.kl_loss_zc = loss_kl_zc.item()
        self.kl_loss_za = loss_kl_za.item()

        loss_G += loss_kl_zc + loss_kl_za

        # classification loss on the attribute latent space
        loss_E_cls_self = self.cls_loss(self.E_pred_cls, self.c_org) * self.opts.lambda_cls_E
        self.E_cls_self_loss = loss_E_cls_self.item()

        # regression loss on E_a
        if self.opts.regression:
            loss_E_reg_self = self.reg_loss(self.E_pred_reg, self.c_reg) * self.opts.lambda_cls_E
            self.E_reg_self_loss = loss_E_reg_self.item()
            loss_E_cls_self += loss_E_reg_self

        self.E_cls_loss = loss_E_cls_self.item()
        loss_G += loss_E_cls_self

        # feature attribution map loss
        if self.opts.loss_diff_M:
            diff_M_reg_loss = torch.abs(torch.cat((self.diff_fake_A_encoded,
                                                   self.diff_fake_B_encoded),0)).mean() * self.opts.lambda_diff_M_reg
            self.diff_M_loss = diff_M_reg_loss.item()
            loss_G += diff_M_reg_loss

        # retain graph for backward_G_alone
        loss_G.backward(retain_graph=True)
        self.G_loss = loss_G.item()

    def backward_E_content(self, z_content):
        """
        Content encoder (E_c) losses using the content discriminator
        :param z_content: content latent vector
        :return:
        """
        # Update encoder to fool discriminator
        pred_cls = self.disContent.forward(z_content, mode='cls')
        if not (self.opts.D_content_dis_cls_all1 == 0):
            # the goal is to learn all classes == 0.5
            all1 = self.opts.D_content_dis_cls_all1 * torch.ones_like(self.c_org).to(self.device)
            loss_E_content = self.cls_loss(pred_cls, all1) * self.opts.lambda_E_content_cls
        else:
            # the goal is to fool discriminator- i.e. reverse the classes
            loss_E_content = self.cls_loss(pred_cls, 1 - self.c_org) * self.opts.lambda_E_content_cls
        return loss_E_content

    def backward_G_alone(self):
        """
        Accumulate all losses on fake images for generator, backward
        :return:
        """
        pred_fake, pred_fake_cls = self.dis2.forward(self.fake_random_img)
        loss_G_GAN2 = 0
        for out_a in pred_fake:
            outputs_fake = torch.sigmoid(out_a)
            all_ones = torch.ones_like(outputs_fake).to(self.device)
            loss_G_GAN2 += nn.functional.binary_cross_entropy(outputs_fake, all_ones)

        # classification
        loss_G_cls2 = self.cls_loss(pred_fake_cls, self.c_org) * self.opts.lambda_cls_G
        self.G_gan2_cls_loss = loss_G_cls2.item()

        loss_G_GAN2 = self.opts.lambda_G_gan * loss_G_GAN2
        loss_G = loss_G_GAN2 + loss_G_cls2
        self.G_gan2_loss = loss_G_GAN2.item()
        self.G_gan2_cls_loss = loss_G_cls2.item()

        # latent regression loss
        if self.opts.loss_latent_l1_random:
            loss_z_L1_a = torch.mean(torch.abs(self.mu2_a - self.z_random_a)) * self.opts.lambda_latent_l1
            loss_z_L1_b = torch.mean(torch.abs(self.mu2_b - self.z_random_b)) * self.opts.lambda_latent_l1
            self.l1_recon_random_z_loss = loss_z_L1_a.item() + loss_z_L1_b.item()
            loss_z_L1 = loss_z_L1_a + loss_z_L1_b
            loss = loss_G + loss_z_L1
            loss.backward()
        else:
            loss_G.backward()

    def _l2_regularize(self, mu):
        """
        l2 regularization on weights
        :param mu:
        :return:
        """
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def assemble_outputs(self):
        """
        Assesmble images to be saved for 2D data
        :return:
        """
        images_a = (self.real_A).detach()
        images_b = (self.real_B).detach()
        images_a1 = (self.fake_B_encoded).detach()
        images_a2 = (self.fake_B_random).detach()
        images_a3 = (self.diff_fake_B_encoded).detach()
        images_a4 = (self.fake_AA_encoded).detach()
        images_a5 = (self.fake_A_recon).detach()
        images_b1 = (self.fake_A_encoded).detach()
        images_b2 = (self.fake_A_random).detach()
        images_b3 = (self.diff_fake_A_encoded).detach()
        images_b4 = (self.fake_BB_encoded).detach()
        images_b5 = (self.fake_B_recon).detach()
        if not self.mask is None:
            mask_a = (self.mask_a.unsqueeze(1)).detach()
            mask_b = (self.mask_b.unsqueeze(1)).detach()
            row1 = torch.cat(
            (images_a[0:1, ::], mask_a[0:1, ::], images_a1[0:1, ::], images_a2[0:1, ::], images_a3[0:1, ::], images_a4[0:1, ::], images_a5[0:1, ::]), 3)
            row2 = torch.cat(
            (images_b[0:1, ::], mask_b[0:1, ::], images_b1[0:1, ::], images_b2[0:1, ::], images_b3[0:1, ::], images_b4[0:1, ::], images_b5[0:1, ::]), 3)
        else:
            row1 = torch.cat(
            (images_a[0:1, ::], images_a1[0:1, ::], images_a2[0:1, ::], images_a3[0:1, ::], images_a4[0:1, ::], images_a5[0:1, ::]), 3)
            row2 = torch.cat(
            (images_b[0:1, ::], images_b1[0:1, ::], images_b2[0:1, ::], images_b3[0:1, ::], images_b4[0:1, ::], images_b5[0:1, ::]), 3)

        attr_row = None

        images_a_content = torch.mean(self.z_content_a, dim=1, keepdim=True)
        images_b_content = torch.mean(self.z_content_b, dim=1, keepdim=True)
        content_row = torch.cat((images_a_content[0:1, ::], images_b_content[0:1, ::]), 3)

        return torch.cat((row1, row2), 2), content_row, attr_row

    def assemble_outputs_3d(self):
        """
        Assesmble images to be saved for 3D data
        :return:
        """
        images_a = self._normalize_image(self.real_A).detach().cpu().numpy()[0, 0, ::]
        images_b = self._normalize_image(self.real_B).detach().cpu().numpy()[0, 0, ::]
        images_a1 = self._normalize_image(self.fake_AA_encoded).detach().cpu().numpy()[0, 0, ::]
        images_b1 = self._normalize_image(self.fake_BB_encoded).detach().cpu().numpy()[0, 0, ::]
        images_a_clc = self._normalize_image(self.fake_A_recon).detach().cpu().numpy()[0, 0, ::]
        images_b_clc = self._normalize_image(self.fake_B_recon).detach().cpu().numpy()[0, 0, ::]
        images_a_random = self._normalize_image(self.fake_B_random).detach().cpu().numpy()[0, 0, ::]
        images_b_random = self._normalize_image(self.fake_A_random).detach().cpu().numpy()[0, 0, ::]
        if not self.mask is None:
            mask_a = self._normalize_image(self.mask_a).detach().cpu().numpy()[0, 0, ::]
            mask_b = self._normalize_image(self.mask_b).detach().cpu().numpy()[0, 0, ::]
        else:
            mask_a = None
            mask_b = None

        if self.opts.nz == 640:
            images_a_attr = self.z_attr_a.view(self.z_attr_a.size(0), 1, 8, 10, 8)
            images_b_attr = self.z_attr_b.view(self.z_attr_b.size(0), 1, 8, 10, 8)
            images_a_attr = self._normalize_image(images_a_attr).detach().cpu().numpy()[0, 0, ::]
            images_b_attr = self._normalize_image(images_b_attr).detach().cpu().numpy()[0, 0, ::]
        elif self.opts.nz == 64:
            images_a_attr = self.z_attr_a.view(self.z_attr_a.size(0), 1, 8, 8)
            images_b_attr = self.z_attr_b.view(self.z_attr_b.size(0), 1, 8, 8)
            images_a_attr = self._normalize_image(images_a_attr).detach().cpu().numpy()[0, 0, ::]
            images_b_attr = self._normalize_image(images_b_attr).detach().cpu().numpy()[0, 0, ::]
        else:
            images_a_attr = None
            images_b_attr = None

        images_a_content = self._normalize_image(torch.mean(self.z_content_a, dim=1, keepdim=True)).detach().cpu().numpy()[0, 0, ::]
        images_b_content = self._normalize_image(torch.mean(self.z_content_b, dim=1, keepdim=True)).detach().cpu().numpy()[0, 0, ::]

        images_a2 = self._normalize_image(self.fake_B_encoded).detach().cpu().numpy()[0, 0, ::]
        images_a3 = self._normalize_image(self.diff_fake_B_encoded).detach().cpu().numpy()[0, 0, ::]
        images_b2 = self._normalize_image(self.fake_A_encoded).detach().cpu().numpy()[0, 0, ::]
        images_b3 = self._normalize_image(self.diff_fake_A_encoded).detach().cpu().numpy()[0, 0, ::]

        return images_a, images_b, images_a1, images_a2, images_a3, images_b1, images_b2, images_b3, images_a_content, \
               images_b_content, images_a_attr, images_b_attr,\
               images_a_clc, images_b_clc, images_a_random, images_b_random, mask_a, mask_b

    def _normalize_image(self, x):
        return x[:, 0:3, :, :]

    def save(self, filename, ep, total_it, it):
        """
        Save networks
        :param filename: save path
        :param ep: current epoch
        :param total_it: total iterations
        :param it: current iteration in epoch
        :return:
        """
        state = {}
        state['ep'] = ep
        state['total_it'] = total_it
        state['it'] = it
        state['enc_c'] = self.enc_c.state_dict()
        state['enc_a'] = self.enc_a.state_dict()
        state['enc_c_opt'] = self.enc_c_opt.state_dict()
        state['enc_c_opt'] = self.enc_a_opt.state_dict()
        state['disContent'] = self.disContent.state_dict()
        state['disContent_opt'] = self.disContent_opt.state_dict()
        state['gen'] = self.gen.state_dict()
        state['gen_opt'] = self.gen_opt.state_dict()
        state['dis1'] = self.dis1.state_dict()
        state['dis2'] = self.dis2.state_dict()
        state['dis1_opt'] = self.dis1_opt.state_dict()
        state['dis2_opt'] = self.dis2_opt.state_dict()
        torch.save(state, filename)
        return

    def resume(self, model_dir, device_0, device_1, train=True):
        """
        Load network states
        :param model_dir: load path
        :param device_0: original gpu device
        :param device_1: gpu device to use
        :param train: whether to train or test
        :return: current epoch, total iterations, current iteration
        """
        checkpoint = torch.load(model_dir, map_location={device_0: device_1})
        if train:
            self.dis1.load_state_dict(checkpoint['dis1'], strict=False)
            self.dis2.load_state_dict(checkpoint['dis2'], strict=False)
            self.disContent.load_state_dict(checkpoint['disContent'], strict=False)
        self.enc_c.load_state_dict(checkpoint['enc_c'], strict=False)
        self.enc_a.load_state_dict(checkpoint['enc_a'], strict=False)
        self.gen.load_state_dict(checkpoint['gen'], strict=False)

        try:
            it = checkpoint['it']
        except:
            it = 0
        return checkpoint['ep'], checkpoint['total_it'], it

    def test_forward_random_group(self, image, c_org=None, num=50):
        """
        Method for translation from one image of one class to another class.
        Using rejection sampling - this will give the mean and variance maps.
        :param image: image input
        :param c_org: label of image
        :param num: number of times to sample from attribute latent space (for mean and variance maps)
        :return:
        """
        z_content = self.enc_c.forward(image)

        if len(image.size()) == 5:
            output = torch.zeros((num, image.size(1), image.size(2), image.size(3), image.size(4)))
            diff_m_pos = torch.zeros((num, image.size(1), image.size(2), image.size(3), image.size(4)))
            diff_m_neg = torch.zeros((num, image.size(1), image.size(2), image.size(3), image.size(4)))
        else:
            output = torch.zeros((num, image.size(1), image.size(2), image.size(3)))
            diff_m_pos = torch.zeros((num, image.size(1), image.size(2), image.size(3)))
            diff_m_neg = torch.zeros((num, image.size(1), image.size(2), image.size(3)))

        output = output.to(self.device)
        diff_m_pos = diff_m_pos.to(self.device)
        diff_m_neg = diff_m_neg.to(self.device)
        z_random = torch.zeros((num, self.nz)).to(self.device)
        flag = True
        i = 0
        k = 0
        if c_org[0, 0] == 1:
            class_num = 1
        elif c_org[0, 1] == 1:
            class_num = 0
        while flag:
            k = k + 1
            z_random_temp = self._get_z_random(1, self.nz, 'gauss')
            _, _, pred_random, _ = self.enc_a.forward(x=None, z=z_random_temp.detach())
            prob, pred_ind = torch.max(pred_random, 1)
            if (pred_ind == class_num) and (i < num) and (prob > 0.9):
                z_random[i] = z_random_temp
                i = i + 1
            if i == num - 1:
                flag = False
            elif k > int(300 * 2 * 50):
                z_random = self._get_z_random(num, self.nz, 'gauss')
                print('Random z not separable')
                flag = False

        c_inv = 1 - c_org

        for z in range(num):
            z_temp = z_random[z].unsqueeze(0)
            output[z] = self.gen.forward(x=z_content, z=z_temp, c=c_inv)

            diff_m = (output[z].unsqueeze(0) - image)
            diff_m_pos[z] = diff_m
            diff_m_neg[z] = -diff_m

        output = torch.mean(output, dim=0, keepdim=True)
        diff_m_pos_mean = torch.mean(diff_m_pos, dim=0, keepdim=True)
        diff_m_neg_mean = torch.mean(diff_m_neg, dim=0, keepdim=True)
        diff_m_pos_std = torch.std(diff_m_pos, dim=0, keepdim=True)
        diff_m_neg_std = -diff_m_pos_std
        return output, diff_m_pos_mean, diff_m_neg_mean, diff_m_pos_std, diff_m_neg_std

    def test_interpolation(self, image, c_org=None):
        """
        Method for interpolating between 2 input images
        :param image: input image - should be batch size of 2 for the 2 images for interpolation
        :param c_org: label of images
        :return:
        """
        half_size = image.size(0) // 2
        image_a, image_b = torch.split(image, half_size, dim=0)
        z_content = self.enc_c.forward(image)
        z_content_a, z_content_b = torch.split(z_content, half_size, dim=0)
        c_org_a, c_org_b = torch.split(c_org, half_size, dim=0)
        mu, logvar, _, _ = self.enc_a.forward(image)
        std = logvar.mul(0.5).exp_()
        eps = self._get_z_random(std.size(0), std.size(1), 'gauss')
        z_attr = eps.mul(std).add_(mu)
        z_attr_a, z_attr_b = torch.split(z_attr, half_size, dim=0)

        # image transition
        num_interpolation = 10
        temp = torch.FloatTensor(half_size, self.nz)
        temp.copy_(z_attr_a)
        dz = (z_attr_b - z_attr_a) / num_interpolation
        z = torch.FloatTensor(num_interpolation, self.nz)
        for i in range(num_interpolation):
            temp[:, :] = z_attr_a[:, :] + i * dz[:, :]
            z[i, :] = temp
        z = z.to(self.device)

        outputs_a = []
        diff_map_a_pos = []
        diff_map_a_neg = []
        class_pred = []
        reg_pred = []
        for z_temp in z:
            z_temp = z_temp.unsqueeze(0)
            _, _, cls, reg = self.enc_a.forward(image, z=z_temp)
            _, cls = torch.max(cls, 1)
            output = self.gen.forward(x=z_content_a, z=z_temp, c=c_org_b)
            outputs_a.append(output)
            diff_map = output - image_a
            diff_map_a_pos.append(diff_map)
            diff_map_a_neg.append(-diff_map)
            class_pred.append(cls.cpu().numpy())
            reg_pred.append(reg.cpu().numpy())

        num_interpolation = 10
        temp = torch.FloatTensor(half_size, self.nz)
        temp.copy_(z_attr_b)
        dz = (z_attr_a - z_attr_b) / num_interpolation
        z = torch.FloatTensor(num_interpolation, self.nz)
        for i in range(num_interpolation):
            temp[:, :] = z_attr_b[:, :] + i * dz[:, :]
            z[i, :] = temp
        z = z.to(self.device)

        outputs_b = []
        diff_map_b_pos = []
        diff_map_b_neg = []
        i = 0
        for z_temp in z:
            z_temp = z_temp.unsqueeze(0)
            if i == 0:
                _, _, cls, reg = self.enc_a.forward(image, z=z_temp)
                _, cls = torch.max(cls, 1)
                class_pred.append(cls.cpu().numpy())
                reg_pred.append(reg.cpu().numpy())
            output = self.gen.forward(x=z_content_b, z=z_temp, c=c_org_a)
            outputs_b.append(output)
            diff_map = output - image_b
            diff_map_b_pos.append(diff_map)
            diff_map_b_neg.append(-diff_map)
            i = i + 1

        return outputs_a, diff_map_a_pos, diff_map_a_neg, outputs_b, diff_map_b_pos, diff_map_b_neg, class_pred, reg_pred

    def test_forward_transfer(self, image, c_org):
        """
        Method for translating between 2 input images
        :param image: input images
        :param c_org: corresponding labels of the images
        :return:
        """
        half_size = image.size(0) // 2
        z_content = self.enc_c.forward(image)
        mu, logvar, _, _ = self.enc_a.forward(image)
        std = logvar.mul(0.5).exp_()
        eps = self._get_z_random(std.size(0), std.size(1), 'gauss')
        z_attr = eps.mul(std).add_(mu)

        z_content_a, z_content_b = torch.split(z_content, half_size, dim=0)
        content = torch.cat((z_content_b, z_content_a), 0)

        output = self.gen.forward(x=content, z=z_attr, c=c_org)
        output_b, output_a = torch.split(output, half_size, dim=0)
        input_a, input_b = torch.split(image, half_size, dim=0)

        diff_a = output_a - input_a
        diff_b = output_b - input_b

        return output_a, diff_a, -diff_a, output_b, diff_b, -diff_b
