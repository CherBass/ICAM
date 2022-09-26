##################################################
# Author: {Cher Bass}
# Copyright: Copyright {2020}, {ICAM}
# License: {MIT license}
# Credits: {Hsin-Ying Lee}, {2019}, {https://github.com/HsinYingLee/MDMM}
##################################################
from dataloader_utils import *
from options import TrainOptions
from model import ICAM
from matplotlib import pyplot as plt
import time
import os
import json
import torchvision
import numpy as np
import torch
from torch.utils.data import ConcatDataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, mean_absolute_error, \
    mean_squared_error

torch.autograd.set_detect_anomaly(True)
RANDOM_SEED = 8
KFOLDS = 5
IMAGE_SIZE = 128
LATENT_3D = 640
LATENT_2D = 64
RESIZE_IMAGE = True
RESIZE_SIZE_3D = (128, 160, 128)
RESIZE_SIZE_2D = (128, 128)
AGE_RANGE_0 = (40,65)
AGE_RANGE_1 = (65,90)


def main():
    global val_accuracy, val_f1, val_precision, val_recall, val_cross_corr_a, val_cross_corr_b, val_mse, val_mae, \
        saver, ep, save_opts, total_it, iter_counter, t0

    # initialise params
    parser = TrainOptions()
    opts = parser.parse()
    opts.random_seed = RANDOM_SEED
    opts.device = opts.device if torch.cuda.is_available() and opts.gpu else 'cpu'
    opts.name = opts.data_type + '_' + time.strftime("%d%m%Y-%H%M")
    opts.results_path = os.path.join(opts.result_dir, opts.name)
    opts.image_size = IMAGE_SIZE
    opts.age_range_0 = AGE_RANGE_0
    opts.age_range_1 = AGE_RANGE_1
    opts.resize_image = RESIZE_IMAGE
    if opts.data_dim == '2d':
        opts.resize_size = RESIZE_SIZE_2D
    elif opts.data_dim == '3d':
        opts.resize_size = RESIZE_SIZE_3D
    opts.cross_val_folds = KFOLDS
    ep0 = 0
    total_it = 0
    val_accuracy = np.zeros(opts.n_ep)
    val_f1 = np.zeros(opts.n_ep)
    val_precision = np.zeros(opts.n_ep)
    val_recall = np.zeros(opts.n_ep)
    val_cross_corr_a = np.zeros(opts.n_ep)
    val_cross_corr_b = np.zeros(opts.n_ep)
    val_mse = np.zeros(opts.n_ep)
    val_mae = np.zeros(opts.n_ep)
    t0 = time.time()

    # saver for display and output
    if opts.data_dim == '3d':
        from saver_3d import Saver
        opts.nz = LATENT_3D
    else:
        from saver import Saver
        opts.nz = LATENT_2D

    print('\n--- load dataset ---')
    # can add option for new dataloaders here
    # dataloaders for data without cross validation 
    if opts.data_type == 'syn2d':
        healthy_dataloader, healthy_val_dataloader, healthy_test_dataloader, \
        anomaly_dataloader, anomaly_val_dataloader, anomaly_test_dataloader = _load_dataloader(opts)
    elif opts.data_type == 'biobank_age':
        healthy_dataloader, healthy_val_dataloader, healthy_test_dataloader, \
        anomaly_dataloader, anomaly_val_dataloader, anomaly_test_dataloader = init_biobank_age_dataloader(opts)
    elif opts.data_type == 'dhcp_2d':
        healthy_dataloader, healthy_val_dataloader, healthy_test_dataloader, \
        anomaly_dataloader, anomaly_val_dataloader, anomaly_test_dataloader = init_dhcp_dataloader_2d(opts)
        
    # dataloaders for cross validation
    elif opts.data_type == 'syn2d_crossval':
        dataset_train_healthy, healthy_test_dataloader, \
        dataset_train_anomaly, anomaly_test_dataloader = _load_dataloader(opts)
    elif opts.data_type == 'biobank_age_crossval':
        dataset_train_healthy, healthy_test_dataloader, \
        dataset_train_anomaly, anomaly_test_dataloader = init_biobank_age_dataloader_crossval(opts)
    elif opts.data_type == 'dhcp_2d_crossval':
        dataset_train_healthy, healthy_test_dataloader, \
        dataset_train_anomaly, anomaly_test_dataloader = init_dhcp_dataloader_2d_crossval(opts)

    # =========================================================================
    # # train without cross-validation
    # =========================================================================
    if (opts.cross_validation == False):
        print('\n--- load model ---')
        model = ICAM(opts)
        model.setgpu(opts.device)
        model.initialize()
        model.set_scheduler(opts, last_ep=ep0)
        save_opts = vars(opts)
        saver = Saver(opts)
        
        if not os.path.exists(opts.results_path):
            os.makedirs(opts.results_path)
        
        with open(opts.results_path + '/parameters.json', 'w') as file:
            json.dump(save_opts, file, indent=4, sort_keys=True)
        
        print('\n--- train ---')
        for ep in range(ep0, opts.n_ep):
            healthy_data_iter = iter(healthy_dataloader)
            anomaly_data_iter = iter(anomaly_dataloader)
            iter_counter = 0
        
            while iter_counter < len(anomaly_dataloader) and iter_counter < len(healthy_dataloader):
                # output of iter dataloader: [tensor image, tensor label (regression), tensor mask]
                healthy_images, healthy_label_reg, healthy_mask = healthy_data_iter.next()
                anomaly_images, anomaly_label_reg, anomaly_mask = anomaly_data_iter.next()
                healthy_c_org = torch.zeros((healthy_images.size(0), opts.num_domains)).to(opts.device)
                healthy_c_org[:, 0] = 1
                anomaly_c_org = torch.zeros((healthy_images.size(0), opts.num_domains)).to(opts.device)
                anomaly_c_org[:, 1] = 1
                images = torch.cat((healthy_images, anomaly_images), dim=0).type(torch.FloatTensor)
                c_org = torch.cat((healthy_c_org, anomaly_c_org), dim=0).type(torch.FloatTensor)
                label_reg = torch.cat((healthy_label_reg, anomaly_label_reg), dim=0).type(torch.FloatTensor)
        
                if len(healthy_mask.size()) > 2:
                    mask = torch.cat((healthy_mask, anomaly_mask), dim=0).type(torch.FloatTensor)
                    mask = mask.to(opts.device).detach()
                else:
                    mask = None
        
                iter_counter += 1
                if images.size(0) != opts.batch_size:
                    continue
        
                # input data
                images = images.to(opts.device).detach()
                c_org = c_org.to(opts.device).detach()
                label_reg = label_reg.to(opts.device).detach()
        
                # update model
                if (iter_counter % opts.d_iter) != 0 and iter_counter < len(anomaly_dataloader) - opts.d_iter:
                    model.update_D_content(opts, images, c_org)
                    continue
        
                model.update_D(opts, images, c_org, label_reg, mask=mask)
                model.update_EG(opts)
        
                if ((total_it + 1) % opts.train_print_it) == 0:
                    train_accuracy, train_f1, _, _ = model.classification_scores(images, c_org)
                    if opts.regression:
                        train_mse, train_mae, _ = model.regression(images, label_reg)
                if total_it == 0:
                    saver.write_img(ep, total_it, model)
                elif total_it % opts.display_freq == 0:
                    saver.write_img(ep, total_it, model)
                total_it += 1
        
                # save to tensorboard
                saver.write_display(total_it, model)
        
                time_elapsed = time.time() - t0
                hours, rem = divmod(time_elapsed, 3600)
                minutes, seconds = divmod(rem, 60)
        
                if (total_it % opts.train_print_it) == 0:
                    print('Total it: {:d} (ep {:d}, it {:d}), Accuracy: {:.2f}, F1 score: {:.2f}, '
                          'Elapsed time: {:0>2}:{:0>2}:{:05.2f}'
                          .format(total_it, ep, iter_counter, train_accuracy, train_f1, int(hours), int(minutes), seconds))
        
            # save model
            if ep % opts.model_save_freq == 0:
                saver.write_model(ep, total_it, 0, model, epoch=True)
                saver.write_img(ep, total_it, model)
        
            # example validation
            try:
                _validation(opts, model, healthy_val_dataloader, anomaly_val_dataloader)
            except Exception as e:
                print(f'Encountered error during validation - {e}')
                raise e
        
        # example test
        try:
            _test(opts, model, healthy_test_dataloader, anomaly_test_dataloader)
        except Exception as e:
            print(f'Encountered error during test - {e}')
            raise e
        
        # save last model
        saver.write_model(ep, total_it, iter_counter, model, model_name='model_last')
        saver.write_img(ep, total_it, model)

    # =========================================================================
    # train with cross-validation
    # =========================================================================
    elif (opts.cross_validation == True):
        # For Cross Validation 
        kfold = KFold(n_splits=KFOLDS, shuffle=True, random_state=RANDOM_SEED) #creates 5 folds
        
        # Save final test results across all folds 
        test_results_mae = {}
        test_results_mse = {}
        test_results_acc = {}
        test_results_f1 = {}
        
        # K-fold Cross Validation model evaluation
        for fold, ((train_ids_healthy, val_ids_healthy), (train_ids_anomal, val_ids_anomal) ) in enumerate(zip(kfold.split(dataset_train_healthy), kfold.split(dataset_train_anomaly))):
        
            print('--------------------------------')
            print(f'FOLD {fold}')
            print('--------------------------------')
                    
            train_subsampler_healthy = torch.utils.data.SubsetRandomSampler(train_ids_healthy)
            val_subsampler_healthy = torch.utils.data.SubsetRandomSampler(val_ids_healthy) 
            
            train_subsampler_anomal = torch.utils.data.SubsetRandomSampler(train_ids_anomal)
            val_subsampler_anomal = torch.utils.data.SubsetRandomSampler(val_ids_anomal) 
            
            print('Train Healthy len subset: ' + str(len(train_subsampler_healthy)))
            print('Val Healthy len subset: ' + str(len(val_subsampler_healthy)))
            
            print('Train Anomaly len subset: ' + str(len(train_subsampler_anomal)))
            print('Val Anomaly len subset: ' + str(len(val_subsampler_anomal)))
            
            
            healthy_dataloader = torch.utils.data.DataLoader(dataset_train_healthy, batch_size=2//2,
                                                           sampler=train_subsampler_healthy)
            healthy_val_dataloader = torch.utils.data.DataLoader(dataset_train_healthy, batch_size=2//2,
                                                         sampler=val_subsampler_healthy)
            
            anomaly_dataloader = torch.utils.data.DataLoader(dataset_train_anomaly, batch_size=2//2,
                                                        sampler=train_subsampler_anomal)
            anomaly_val_dataloader = torch.utils.data.DataLoader(dataset_train_anomaly, batch_size=2//2,
                                                         sampler=val_subsampler_anomal)
        
            print('\n--- load model ---')
            model = ICAM(opts)
            model.setgpu(opts.device)
            model.initialize()
            model.set_scheduler(opts, last_ep=ep0)
            save_opts = vars(opts)
            saver = Saver(opts)
        
            if not os.path.exists(opts.results_path):
                os.makedirs(opts.results_path)
        
            with open(opts.results_path + '/parameters_fold' + str(fold) + '.json', 'w') as file:
                json.dump(save_opts, file, indent=4, sort_keys=True)
        
            print('\n--- train ---')
            for ep in range(ep0, opts.n_ep):
                healthy_data_iter = iter(healthy_dataloader)
                anomaly_data_iter = iter(anomaly_dataloader)
                iter_counter = 0
        
                while iter_counter < len(anomaly_dataloader) and iter_counter < len(healthy_dataloader):
                    # output of iter dataloader: [tensor image, tensor label (regression), tensor mask]
                    healthy_images, healthy_label_reg, healthy_mask = healthy_data_iter.next()
                    anomaly_images, anomaly_label_reg, anomaly_mask = anomaly_data_iter.next()
                    healthy_c_org = torch.zeros((healthy_images.size(0), opts.num_domains)).to(opts.device)
                    healthy_c_org[:, 0] = 1
                    anomaly_c_org = torch.zeros((healthy_images.size(0), opts.num_domains)).to(opts.device)
                    anomaly_c_org[:, 1] = 1
                    images = torch.cat((healthy_images, anomaly_images), dim=0).type(torch.FloatTensor)
                    c_org = torch.cat((healthy_c_org, anomaly_c_org), dim=0).type(torch.FloatTensor)
                    label_reg = torch.cat((healthy_label_reg, anomaly_label_reg), dim=0).type(torch.FloatTensor)
        
                    if len(healthy_mask.size()) > 2:
                        mask = torch.cat((healthy_mask, anomaly_mask), dim=0).type(torch.FloatTensor)
                        mask = mask.to(opts.device).detach()
                    else:
                        mask = None
        
                    iter_counter += 1
                    if images.size(0) != opts.batch_size:
                        continue
        
                    # input data
                    images = images.to(opts.device).detach()
                    c_org = c_org.to(opts.device).detach()
                    label_reg = label_reg.to(opts.device).detach()
        
                    # update model
                    if (iter_counter % opts.d_iter) != 0 and iter_counter < len(anomaly_dataloader) - opts.d_iter:
                        model.update_D_content(opts, images, c_org)
                        continue
        
                    model.update_D(opts, images, c_org, label_reg, mask=mask)
                    model.update_EG(opts)
        
                    if ((total_it + 1) % opts.train_print_it) == 0:
                        train_accuracy, train_f1, _, _ = model.classification_scores(images, c_org)
                        if opts.regression:
                            train_mse, train_mae, _ = model.regression(images, label_reg)
                    if total_it == 0:
                        saver.write_img(ep, total_it, model)
                    elif total_it % opts.display_freq == 0:
                        saver.write_img(ep, total_it, model)
                    total_it += 1
        
                    # save to tensorboard
                    saver.write_display(total_it, model)
        
                    time_elapsed = time.time() - t0
                    hours, rem = divmod(time_elapsed, 3600)
                    minutes, seconds = divmod(rem, 60)
        
                    if (total_it % opts.train_print_it) == 0:
                        print('Total it: {:d} (ep {:d}, it {:d}), Accuracy: {:.2f}, F1 score: {:.2f}, '
                              'Elapsed time: {:0>2}:{:0>2}:{:05.2f}'
                              .format(total_it, ep, iter_counter, train_accuracy, train_f1, int(hours), int(minutes), seconds))
                    
                # Validation - each epoch during training fold
                print('Performing validation inside fold.....')
                try:
                    mae_val, mse_val, acc_val, f1_val =  _validation_crossval(opts, model, healthy_val_dataloader, anomaly_val_dataloader, fold)
                except Exception as e:
                    print(f'Encountered error during validation - {e}')
                    raise e
        
                # Save model end of fold
                if ep % opts.model_save_freq == 0:
                    saver.write_model(ep, total_it, 0, model, epoch=True)
                    saver.write_img(ep, total_it, model)
        
        
            print('Ended training in fold - starting test with hold-out data.....')    
            # Test - using hold-out test set at the end of training fold
            try:
                mae_test, mse_test, acc_test, f1_test = _test_crossval(opts, model, healthy_test_dataloader, anomaly_test_dataloader, fold)
            except Exception as e:
                print(f'Encountered error during test - {e}')
                raise e
            
            # Save test results
            test_results_mae[fold] = mae_test
            test_results_mse[fold] = mse_test
            test_results_acc[fold] = acc_test
            test_results_f1[fold] = f1_test
        
            # save last model for fold
            saver.write_model(ep, total_it, iter_counter, model, model_name='model_last_' + str(fold))
            saver.write_img(ep, total_it, model)
        
        
        # ------- Print all fold test results ----------------------------------------------
        print(f'K-FOLD TEST RESULTS FOR {KFOLDS} FOLDS')
        print('--------------------------------')
            
        # RESULTS TEST
        sum = 0.0
        list_values = []
        for key, value in test_results_mae.items():
            print(f'Fold {key}: {value} %')
            sum += value
            list_values.append(value)
        print(f'Average Test MAE: {sum/len(test_results_mae.items())} %')
        std_dev_mae = np.std(list_values)
        print(f'with std deviation MAE: {std_dev_mae} %')
        
        sum = 0.0
        list_values = []
        for key, value in test_results_mse.items():
            print(f'Fold {key}: {value} %')
            sum += value
            list_values.append(value)
        print(f'Average Test MSE: {sum/len(test_results_mse.items())} %')
        std_dev_mse =  np.std(list_values)
        print(f'with std deviation MSE: {std_dev_mse} %')
        
        sum = 0.0
        list_values = []
        for key, value in test_results_acc.items():
            print(f'Fold {key}: {value} %')
            sum += value
            list_values.append(value)
        print(f'Average Test Accuracy: {sum/len(test_results_acc.items())} %')
        std_dev_acc =  np.std(list_values)
        print(f'with std deviation Acc: {std_dev_acc} %')
        
        sum = 0.0
        list_values = []
        for key, value in test_results_f1.items():
            print(f'Fold {key}: {value} %')
            sum += value
            list_values.append(value)
        print(f'Average Test F1: {sum/len(test_results_f1.items())} %')
        std_dev_f1 =  np.std(list_values)
        print(f'with std deviation F1: {std_dev_f1} %')

    return


def _load_dataloader(opts):
    """
    Load correct dataloader based on options.
    2D init_synth_dataloader() is used as default.
    3D dataloader init_biobank_age_dataloader() is shown as an example, but data will need to be acquired.
    :param opts: options
    :return: train, val and test dataloaders for healthy and anomaly datasets
    """
    if opts.data_type == 'syn2d':
        healthy_dataloader = init_synth_dataloader(
            opts, anomaly=False, mode='train', batch_size=opts.batch_size // 2)
        anomaly_dataloader = init_synth_dataloader(
            opts, anomaly=True, mode='train', batch_size=opts.batch_size // 2)
        healthy_val_dataloader = init_synth_dataloader(
            opts, anomaly=False, mode='val', batch_size=opts.val_batch_size // 2)
        anomaly_val_dataloader = init_synth_dataloader(
            opts, anomaly=True, mode='val', batch_size=opts.val_batch_size // 2)
        healthy_test_dataloader = init_synth_dataloader(
            opts, anomaly=False, mode='test', batch_size=opts.val_batch_size // 2)
        anomaly_test_dataloader = init_synth_dataloader(
            opts, anomaly=True, mode='test', batch_size=opts.val_batch_size // 2)
    elif opts.data_type == 'syn2d_crossval':
        healthy_dataloader = init_synth_dataloader_crossval(
            opts, anomaly=False, mode='train', batch_size=opts.val_batch_size // 2)
        anomaly_dataloader = init_synth_dataloader_crossval(
            opts, anomaly=True, mode='train', batch_size=opts.val_batch_size // 2)
        healthy_test_dataloader = init_synth_dataloader_crossval(
            opts, anomaly=False, mode='test', batch_size=opts.val_batch_size // 2)
        anomaly_test_dataloader = init_synth_dataloader_crossval(
            opts, anomaly=True, mode='test', batch_size=opts.val_batch_size // 2)
    elif opts.data_type == 'biobank_age':
        healthy_dataloader, healthy_val_dataloader, healthy_test_dataloader, \
        anomaly_dataloader, anomaly_val_dataloader, anomaly_test_dataloader = init_biobank_age_dataloader(
            opts)

    if opts.cross_validation == False:
    	return healthy_dataloader, healthy_val_dataloader, healthy_test_dataloader, \
           anomaly_dataloader, anomaly_val_dataloader, anomaly_test_dataloader
           
    else:
    	return healthy_dataloader, healthy_test_dataloader, \
           anomaly_dataloader, anomaly_test_dataloader
     


def _validation(opts, model, healthy_val_dataloader, anomaly_val_dataloader):
    """
    Validation function for classification and regression
    :param opts:
    :param model: networks
    :param healthy_test_dataloader:
    :param anomaly_test_dataloader:
    :return:
    """
    e = np.arange(opts.n_ep)
    val_pred_temp = np.zeros((0))
    val_labels = np.zeros((0))
    if opts.regression:
        val_reg_pred_temp = np.zeros((0))
        val_reg_labels = np.zeros((0))
    if opts.cross_corr:
        val_cross_corr_temp_a = np.zeros((0))
        val_cross_corr_temp_b = np.zeros((0))
    healthy_val_iter = iter(healthy_val_dataloader)
    anomaly_val_iter = iter(anomaly_val_dataloader)

    # anomaly dataset should be the same or smaller size than healthy
    anomaly_val_dataloader_len = len(anomaly_val_dataloader)
    healthy_val_dataloader_len = len(healthy_val_dataloader)

    if anomaly_val_dataloader_len > healthy_val_dataloader_len:
        raise Exception(f'anomaly dataloader len {anomaly_val_dataloader_len} is bigger than healthy dataloader'
                        f' len {healthy_val_dataloader_len}')

    for j in range(healthy_val_dataloader_len):
        if j < anomaly_val_dataloader_len:
            healthy_val_images, reg_label_healthy, _ = healthy_val_iter.next()
            anomaly_val_images, reg_label_anomaly, mask = anomaly_val_iter.next()
            healthy_val_c_org = torch.zeros((healthy_val_images.size(0), opts.num_domains)).to(opts.device)
            healthy_val_c_org[:, 0] = 1
            anomaly_val_c_org = torch.zeros((healthy_val_images.size(0), opts.num_domains)).to(opts.device)
            anomaly_val_c_org[:, 1] = 1
            images_val = torch.cat((healthy_val_images, anomaly_val_images), dim=0).type(torch.FloatTensor)
            c_org_val = torch.cat((healthy_val_c_org, anomaly_val_c_org), dim=0).type(torch.FloatTensor)
            reg_val = torch.cat((reg_label_healthy[0], reg_label_anomaly[0]), dim=0).type(torch.FloatTensor)

        else:
            healthy_val_images, reg_label_healthy, _ = healthy_val_iter.next()
            healthy_val_c_org = torch.zeros((healthy_val_images.size(0), opts.num_domains)).to(opts.device)
            healthy_val_c_org[:, 0] = 1
            images_val = healthy_val_images
            c_org_val = healthy_val_c_org
            reg_val = reg_label_healthy[0].type(torch.FloatTensor)

        images_val = images_val.to(opts.device).detach()
        c_org_val = c_org_val.to(opts.device).detach()
        reg_val = reg_val.to(opts.device).detach()
        mask = mask.to(opts.device).detach()

        _, _, pred, reg_pred = model.enc_a.forward(images_val)
        _, y_pred = torch.max(pred, 1)
        _, labels_temp = torch.max(c_org_val, 1)

        val_pred_temp = np.append(val_pred_temp, y_pred.data.cpu().numpy())
        val_labels = np.append(val_labels, labels_temp.data.cpu().numpy())
        if opts.regression:
            val_reg_pred_temp = np.append(val_reg_pred_temp, reg_pred.data.cpu().numpy())
            val_reg_labels = np.append(val_reg_labels, reg_val.data.cpu().numpy())

        if opts.cross_corr:
            cross_corr_a, cross_corr_b = model.cross_correlation(images_val, mask, c_org_val)
            val_cross_corr_temp_a = np.append(val_cross_corr_temp_a, cross_corr_a)
            val_cross_corr_temp_b = np.append(val_cross_corr_temp_b, cross_corr_b)

    val_accuracy[ep] = accuracy_score(val_pred_temp, val_labels)
    val_f1[ep] = f1_score(val_pred_temp, val_labels, average='macro')
    val_precision[ep] = precision_score(val_pred_temp, val_labels, average='macro')
    val_recall[ep] = recall_score(val_pred_temp, val_labels, average='macro')

    time_elapsed = time.time() - t0
    hours, rem = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)

    print('Total it: {:d} (ep {:d}, it {:d}), Val Accuracy: {:.2f}, '
          'Val F1 score: {:.2f}, Elapsed time: {:0>2}:{:0>2}:{:05.2f}'
          .format(total_it, ep, iter_counter, val_accuracy[ep], val_f1[ep], int(hours), int(minutes), seconds))

    if val_f1[ep] >= np.max(val_f1):
        save_opts['val_accuracy'] = np.max(val_accuracy[ep])
        save_opts['val_f1'] = np.max(val_f1[ep])
        save_opts['val_precision'] = np.max(val_precision[ep])
        save_opts['val_recall'] = np.max(val_recall[ep])

    if opts.regression:
        val_mse[ep] = mean_squared_error(val_reg_labels, val_reg_pred_temp)
        val_mae[ep] = mean_absolute_error(val_reg_labels, val_reg_pred_temp)
        if val_mae[ep] <= np.min(val_mae[:ep + 1]):
            save_opts['val_mse'] = val_mse[ep]
            save_opts['val_mae'] = val_mae[ep]

        print('Total it: {:d} (ep {:d}, it {:d}), Val MAE: {:.2f}, '
              'Val MSE: {:.2f}, Elapsed time: {:0>2}:{:0>2}:{:05.2f}'
              .format(total_it, ep, iter_counter, val_mae[ep], val_mse[ep], int(hours), int(minutes), seconds))

        x, y = line_best_fit(val_reg_labels, val_reg_pred_temp)
        yfit = [x + y * xi for xi in val_reg_labels]
        plt.figure()
        plt.plot(val_reg_labels, val_reg_pred_temp, '+')
        plt.plot(val_reg_labels, yfit, 'k', linewidth=1)
        plt.xlabel('true values')
        plt.ylabel('predicted values')
        plt.title('True vs predicted values plot')
        plt.savefig(opts.results_path + '/val_regression_plot.png')
        plt.close()

    if opts.cross_corr:
        val_cross_corr_a[ep] = np.mean(val_cross_corr_temp_a)
        val_cross_corr_b[ep] = np.mean(val_cross_corr_temp_b)
        print('Total it: {:d} (ep {:d}, it {:d}), Val cross corr a: {:.2f}, '
              'Val cross corr b: {:.2f}, Elapsed time: {:0>2}:{:0>2}:{:05.2f}'
              .format(total_it, ep, iter_counter, val_cross_corr_a[ep], val_cross_corr_b[ep], int(hours), int(minutes),
                      seconds))
        save_opts['val_cross_corr_a'] = np.max(val_cross_corr_a)
        save_opts['val_cross_corr_b'] = np.max(val_cross_corr_b)

    with open(opts.results_path + '/parameters.json', 'w') as file:
        json.dump(save_opts, file, indent=4, sort_keys=True)

    # save and plot results
    _save_best_models(opts, model)
    _plot_results(opts, e)
    if opts.data_dim == '2d':
        _translation_example(opts, model, healthy_val_images, anomaly_val_images, 'val_images')


def _validation_crossval(opts, model, healthy_val_dataloader, anomaly_val_dataloader, fold):
    """
    Validation function for classification and regression
    :param opts:
    :param model: networks
    :param healthy_val_dataloader:
    :param anomaly_val_dataloader:
    :return:
    """
    e = np.arange(opts.n_ep)
    val_pred_temp = np.zeros((0))
    val_labels = np.zeros((0))
    if opts.regression:
        val_reg_pred_temp = np.zeros((0))
        val_reg_labels = np.zeros((0))
    if opts.cross_corr:
        val_cross_corr_temp_a = np.zeros((0))
        val_cross_corr_temp_b = np.zeros((0))
    healthy_val_iter = iter(healthy_val_dataloader)
    anomaly_val_iter = iter(anomaly_val_dataloader)

    # anomaly dataset should be the same or smaller size than healthy
    anomaly_val_dataloader_len = len(anomaly_val_dataloader)
    healthy_val_dataloader_len = len(healthy_val_dataloader)

    if anomaly_val_dataloader_len > healthy_val_dataloader_len:
        raise Exception(f'anomaly dataloader len {anomaly_val_dataloader_len} is bigger than healthy dataloader'
                        f' len {healthy_val_dataloader_len}')

    for j in range(healthy_val_dataloader_len):
        if j < anomaly_val_dataloader_len:
            healthy_val_images, reg_label_healthy, _ = healthy_val_iter.next()
            anomaly_val_images, reg_label_anomaly, mask = anomaly_val_iter.next()
            healthy_val_c_org = torch.zeros((healthy_val_images.size(0), opts.num_domains)).to(opts.device)
            healthy_val_c_org[:, 0] = 1
            anomaly_val_c_org = torch.zeros((healthy_val_images.size(0), opts.num_domains)).to(opts.device)
            anomaly_val_c_org[:, 1] = 1
            images_val = torch.cat((healthy_val_images, anomaly_val_images), dim=0).type(torch.FloatTensor)
            c_org_val = torch.cat((healthy_val_c_org, anomaly_val_c_org), dim=0).type(torch.FloatTensor)
            reg_val = torch.cat((reg_label_healthy[0], reg_label_anomaly[0]), dim=0).type(torch.FloatTensor)

        else:
            healthy_val_images, reg_label_healthy, _ = healthy_val_iter.next()
            healthy_val_c_org = torch.zeros((healthy_val_images.size(0), opts.num_domains)).to(opts.device)
            healthy_val_c_org[:, 0] = 1
            images_val = healthy_val_images
            c_org_val = healthy_val_c_org
            reg_val = reg_label_healthy[0].type(torch.FloatTensor)

        images_val = images_val.to(opts.device).detach()
        c_org_val = c_org_val.to(opts.device).detach()
        reg_val = reg_val.to(opts.device).detach()
        mask = mask.to(opts.device).detach()

        _, _, pred, reg_pred = model.enc_a.forward(images_val)
        _, y_pred = torch.max(pred, 1)
        _, labels_temp = torch.max(c_org_val, 1)

        val_pred_temp = np.append(val_pred_temp, y_pred.data.cpu().numpy())
        val_labels = np.append(val_labels, labels_temp.data.cpu().numpy())
        if opts.regression:
            val_reg_pred_temp = np.append(val_reg_pred_temp, reg_pred.data.cpu().numpy())
            val_reg_labels = np.append(val_reg_labels, reg_val.data.cpu().numpy())

        if opts.cross_corr:
            cross_corr_a, cross_corr_b = model.cross_correlation(images_val, mask, c_org_val)
            val_cross_corr_temp_a = np.append(val_cross_corr_temp_a, cross_corr_a)
            val_cross_corr_temp_b = np.append(val_cross_corr_temp_b, cross_corr_b)

    val_accuracy[ep] = accuracy_score(val_pred_temp, val_labels)
    val_f1[ep] = f1_score(val_pred_temp, val_labels, average='macro')
    val_precision[ep] = precision_score(val_pred_temp, val_labels, average='macro')
    val_recall[ep] = recall_score(val_pred_temp, val_labels, average='macro')

    time_elapsed = time.time() - t0
    hours, rem = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)

    print('Total it: {:d} (ep {:d}, it {:d}), Val Accuracy: {:.2f}, '
          'Val F1 score: {:.2f}, Elapsed time: {:0>2}:{:0>2}:{:05.2f}'
          .format(total_it, ep, iter_counter, val_accuracy[ep], val_f1[ep], int(hours), int(minutes), seconds))

    if val_f1[ep] >= np.max(val_f1):
        save_opts['val_accuracy'] = np.max(val_accuracy[ep])
        save_opts['val_f1'] = np.max(val_f1[ep])
        save_opts['val_precision'] = np.max(val_precision[ep])
        save_opts['val_recall'] = np.max(val_recall[ep])

    if opts.regression:
        val_mse[ep] = mean_squared_error(val_reg_labels, val_reg_pred_temp)
        val_mae[ep] = mean_absolute_error(val_reg_labels, val_reg_pred_temp)
        if val_mae[ep] <= np.min(val_mae[:ep + 1]):
            save_opts['val_mse'] = val_mse[ep]
            save_opts['val_mae'] = val_mae[ep]

        print('Total it: {:d} (ep {:d}, it {:d}), Val MAE: {:.2f}, '
              'Val MSE: {:.2f}, Elapsed time: {:0>2}:{:0>2}:{:05.2f}'
              .format(total_it, ep, iter_counter, val_mae[ep], val_mse[ep], int(hours), int(minutes), seconds))

        x, y = line_best_fit(val_reg_labels, val_reg_pred_temp)
        yfit = [x + y * xi for xi in val_reg_labels]
        plt.figure()
        plt.plot(val_reg_labels, val_reg_pred_temp, '+')
        plt.plot(val_reg_labels, yfit, 'k', linewidth=1)
        plt.xlabel('true values')
        plt.ylabel('predicted values')
        plt.title('True vs predicted values plot')
        plt.savefig(opts.results_path + '/val_regression_plot_fold' + str(fold) + '.png')
        plt.close()

    if opts.cross_corr:
        val_cross_corr_a[ep] = np.mean(val_cross_corr_temp_a)
        val_cross_corr_b[ep] = np.mean(val_cross_corr_temp_b)
        print('Total it: {:d} (ep {:d}, it {:d}), Val cross corr a: {:.2f}, '
              'Val cross corr b: {:.2f}, Elapsed time: {:0>2}:{:0>2}:{:05.2f}'
              .format(total_it, ep, iter_counter, val_cross_corr_a[ep], val_cross_corr_b[ep], int(hours), int(minutes),
                      seconds))
        save_opts['val_cross_corr_a'] = np.max(val_cross_corr_a)
        save_opts['val_cross_corr_b'] = np.max(val_cross_corr_b)

    with open(opts.results_path + '/parameters' + str(fold) +'.json', 'w') as file:
        json.dump(save_opts, file, indent=4, sort_keys=True)

    # save and plot results
    _save_best_models(opts, model)
    _plot_results_crossval(opts, e, fold)
    if opts.data_dim == '2d':
        _translation_example(opts, model, healthy_val_images, anomaly_val_images, 'val_images_fold_' + str(fold))
    
    return val_mae[ep], val_mse[ep], val_accuracy[ep], val_f1[ep]


def _test(opts, model, healthy_test_dataloader, anomaly_test_dataloader):
    """
    Testing function for classification and regression with example translation
    :param opts:
    :param model: networks
    :param healthy_test_dataloader:
    :param anomaly_test_dataloader:
    :return:
    """
    val_pred_temp = np.zeros((0))
    val_labels = np.zeros((0))
    if opts.regression:
        val_reg_pred_temp = np.zeros((0))
        val_reg_labels = np.zeros((0))
    if opts.cross_corr:
        val_cross_corr_temp_a = np.zeros((0))
        val_cross_corr_temp_b = np.zeros((0))
    healthy_val_iter = iter(healthy_test_dataloader)
    anomaly_val_iter = iter(anomaly_test_dataloader)

    # anomaly dataset should be the same or smaller size than healthy
    anomaly_val_dataloader_len = len(anomaly_test_dataloader)
    healthy_val_dataloader_len = len(healthy_test_dataloader)

    if anomaly_val_dataloader_len > healthy_val_dataloader_len:
        raise Exception(f'anaomaly dataloader len {anomaly_val_dataloader_len} is bigger than healthy dataloader'
                        f' len {healthy_val_dataloader_len}')

    # anomaly dataset should be the same or smaller size than healthy
    for j in range(healthy_val_dataloader_len):
        if j < anomaly_val_dataloader_len:
            healthy_val_images, reg_label_healthy, _ = healthy_val_iter.next()
            anomaly_val_images, reg_label_anomaly, mask = anomaly_val_iter.next()
            healthy_val_c_org = torch.zeros((healthy_val_images.size(0), opts.num_domains)).to(opts.device)
            healthy_val_c_org[:, 0] = 1
            anomaly_val_c_org = torch.zeros((healthy_val_images.size(0), opts.num_domains)).to(opts.device)
            anomaly_val_c_org[:, 1] = 1
            images_val = torch.cat((healthy_val_images, anomaly_val_images), dim=0).type(torch.FloatTensor)
            c_org_val = torch.cat((healthy_val_c_org, anomaly_val_c_org), dim=0).type(torch.FloatTensor)
            reg_val = torch.cat((reg_label_healthy[0], reg_label_anomaly[0]), dim=0).type(torch.FloatTensor)

        else:
            healthy_val_images, reg_label_healthy, _ = healthy_val_iter.next()
            healthy_val_c_org = torch.zeros((healthy_val_images.size(0), opts.num_domains)).to(opts.device)
            healthy_val_c_org[:, 0] = 1
            images_val = healthy_val_images
            c_org_val = healthy_val_c_org
            reg_val = reg_label_healthy[0].type(torch.FloatTensor)

        images_val = images_val.to(opts.device).detach()
        c_org_val = c_org_val.to(opts.device).detach()
        reg_val = reg_val.to(opts.device).detach()
        mask = mask.to(opts.device).detach()

        _, _, pred, reg_pred = model.enc_a.forward(images_val)
        _, y_pred = torch.max(pred, 1)
        _, labels_temp = torch.max(c_org_val, 1)

        val_pred_temp = np.append(val_pred_temp, y_pred.data.cpu().numpy())
        val_labels = np.append(val_labels, labels_temp.data.cpu().numpy())
        if opts.regression:
            val_reg_pred_temp = np.append(val_reg_pred_temp, reg_pred.data.cpu().numpy())
            val_reg_labels = np.append(val_reg_labels, reg_val.data.cpu().numpy())

        if opts.cross_corr:
            cross_corr_a, cross_corr_b = model.cross_correlation(images_val, mask, c_org_val)
            val_cross_corr_temp_a = np.append(val_cross_corr_temp_a, cross_corr_a)
            val_cross_corr_temp_b = np.append(val_cross_corr_temp_b, cross_corr_b)

    val_accuracy = accuracy_score(val_pred_temp, val_labels)
    val_f1 = f1_score(val_pred_temp, val_labels, average='macro')
    val_precision = precision_score(val_pred_temp, val_labels, average='macro')
    val_recall = recall_score(val_pred_temp, val_labels, average='macro')

    save_opts['test_accuracy'] = val_accuracy
    save_opts['test_f1'] = val_f1
    save_opts['test_precision'] = val_precision
    save_opts['test_recall'] = val_recall

    if opts.regression:
        val_mae = mean_absolute_error(val_reg_labels, val_reg_pred_temp)
        val_mse = mean_squared_error(val_reg_labels, val_reg_pred_temp)

        save_opts['test_mse'] = val_mse
        save_opts['test_mae'] = val_mae

        x, y = line_best_fit(val_reg_labels, val_reg_pred_temp)
        yfit = [x + y * xi for xi in val_reg_labels]
        plt.figure()
        plt.plot(val_reg_labels, val_reg_pred_temp, '+')
        plt.plot(val_reg_labels, yfit, 'k', linewidth=1)
        plt.xlabel('true values')
        plt.ylabel('predicted values')
        plt.title('True vs predicted values plot')
        plt.savefig(opts.results_path + '/test_regression_plot.png')
        plt.close()

    if opts.cross_corr:
        val_cross_corr_a = np.mean(val_cross_corr_temp_a)
        val_cross_corr_b = np.mean(val_cross_corr_temp_b)

        save_opts['test_cross_corr_a'] = val_cross_corr_a
        save_opts['test_cross_corr_b'] = val_cross_corr_b

    with open(opts.results_path + '/test_results.json', 'w') as file:
        json.dump(save_opts, file, indent=4, sort_keys=True)

    # plot translation figures
    if opts.data_dim == '2d':
        _translation_example(opts, model, healthy_val_images, anomaly_val_images, 'test_images')


def _test_crossval(opts, model, healthy_test_dataloader, anomaly_test_dataloader, fold):
    """
    Testing function for classification and regression with example translation
    :param opts:
    :param model: networks
    :param healthy_test_dataloader:
    :param anomaly_test_dataloader:
    :return:
    """
    val_pred_temp = np.zeros((0))
    val_labels = np.zeros((0))
    if opts.regression:
        val_reg_pred_temp = np.zeros((0))
        val_reg_labels = np.zeros((0))
    if opts.cross_corr:
        val_cross_corr_temp_a = np.zeros((0))
        val_cross_corr_temp_b = np.zeros((0))
    healthy_val_iter = iter(healthy_test_dataloader)
    anomaly_val_iter = iter(anomaly_test_dataloader)

    # anomaly dataset should be the same or smaller size than healthy
    anomaly_val_dataloader_len = len(anomaly_test_dataloader)
    healthy_val_dataloader_len = len(healthy_test_dataloader)

    if anomaly_val_dataloader_len > healthy_val_dataloader_len:
        raise Exception(f'anomaly dataloader len {anomaly_val_dataloader_len} is bigger than healthy dataloader'
                        f' len {healthy_val_dataloader_len}')

    # anomaly dataset should be the same or smaller size than healthy
    for j in range(healthy_val_dataloader_len):
        if j < anomaly_val_dataloader_len:
            healthy_val_images, reg_label_healthy, _ = healthy_val_iter.next()
            anomaly_val_images, reg_label_anomaly, mask = anomaly_val_iter.next()
            healthy_val_c_org = torch.zeros((healthy_val_images.size(0), opts.num_domains)).to(opts.device)
            healthy_val_c_org[:, 0] = 1
            anomaly_val_c_org = torch.zeros((healthy_val_images.size(0), opts.num_domains)).to(opts.device)
            anomaly_val_c_org[:, 1] = 1
            images_val = torch.cat((healthy_val_images, anomaly_val_images), dim=0).type(torch.FloatTensor)
            c_org_val = torch.cat((healthy_val_c_org, anomaly_val_c_org), dim=0).type(torch.FloatTensor)
            reg_val = torch.cat((reg_label_healthy[0], reg_label_anomaly[0]), dim=0).type(torch.FloatTensor)

        else:
            healthy_val_images, reg_label_healthy, _ = healthy_val_iter.next()
            healthy_val_c_org = torch.zeros((healthy_val_images.size(0), opts.num_domains)).to(opts.device)
            healthy_val_c_org[:, 0] = 1
            images_val = healthy_val_images
            c_org_val = healthy_val_c_org
            reg_val = reg_label_healthy[0].type(torch.FloatTensor)

        images_val = images_val.to(opts.device).detach()
        c_org_val = c_org_val.to(opts.device).detach()
        reg_val = reg_val.to(opts.device).detach()
        mask = mask.to(opts.device).detach()

        _, _, pred, reg_pred = model.enc_a.forward(images_val)
        _, y_pred = torch.max(pred, 1)
        _, labels_temp = torch.max(c_org_val, 1)

        val_pred_temp = np.append(val_pred_temp, y_pred.data.cpu().numpy())
        val_labels = np.append(val_labels, labels_temp.data.cpu().numpy())
        if opts.regression:
            val_reg_pred_temp = np.append(val_reg_pred_temp, reg_pred.data.cpu().numpy())
            val_reg_labels = np.append(val_reg_labels, reg_val.data.cpu().numpy())

        if opts.cross_corr:
            cross_corr_a, cross_corr_b = model.cross_correlation(images_val, mask, c_org_val)
            val_cross_corr_temp_a = np.append(val_cross_corr_temp_a, cross_corr_a)
            val_cross_corr_temp_b = np.append(val_cross_corr_temp_b, cross_corr_b)

    val_accuracy = accuracy_score(val_pred_temp, val_labels)
    val_f1 = f1_score(val_pred_temp, val_labels, average='macro')
    val_precision = precision_score(val_pred_temp, val_labels, average='macro')
    val_recall = recall_score(val_pred_temp, val_labels, average='macro')

    save_opts['test_accuracy'] = val_accuracy
    save_opts['test_f1'] = val_f1
    save_opts['test_precision'] = val_precision
    save_opts['test_recall'] = val_recall

    if opts.regression:
        val_mae = mean_absolute_error(val_reg_labels, val_reg_pred_temp)
        val_mse = mean_squared_error(val_reg_labels, val_reg_pred_temp)

        save_opts['test_mse'] = val_mse
        save_opts['test_mae'] = val_mae

        x, y = line_best_fit(val_reg_labels, val_reg_pred_temp)
        yfit = [x + y * xi for xi in val_reg_labels]
        plt.figure()
        plt.plot(val_reg_labels, val_reg_pred_temp, '+')
        plt.plot(val_reg_labels, yfit, 'k', linewidth=1)
        plt.xlabel('true values')
        plt.ylabel('predicted values')
        plt.title('True vs predicted values plot')
        plt.savefig(opts.results_path + '/test_regression_plot_fold' + str(fold) + '.png')
        plt.close()

    if opts.cross_corr:
        val_cross_corr_a = np.mean(val_cross_corr_temp_a)
        val_cross_corr_b = np.mean(val_cross_corr_temp_b)

        save_opts['test_cross_corr_a'] = val_cross_corr_a
        save_opts['test_cross_corr_b'] = val_cross_corr_b

    with open(opts.results_path + '/test_results_fold' + str(fold) + '.json', 'w') as file:
        json.dump(save_opts, file, indent=4, sort_keys=True)

    # plot translation figures
    if opts.data_dim == '2d':
        _translation_example(opts, model, healthy_val_images, anomaly_val_images, 'test_images_fold' + str(fold))
    
    if opts.regression:    
        return val_mae, val_mse, val_accuracy, val_f1
    
    elif opts.cross_corr:
        return val_accuracy, val_f1, val_cross_corr_a, val_cross_corr_b
    

def _translation_example(opts, model, healthy_images, anomaly_images, save_name='val_images'):
    """
    Example translation function for 2D inputs only. For 3D inputs, it requires a different saving function.
    :param opts:
    :param model:
    :param healthy_images:
    :param anomaly_images:
    :return:
    """
    path = opts.results_path + '/' + save_name
    if not os.path.exists(path):
        os.makedirs(path)

    # Example usage with anomaly data (i.e. class=1)

    # to achieve translation for anomaly data use label = [0, 1]
    # to achieve translation for healthy data use label = [1, 0]
    c_org_trans = torch.zeros((anomaly_images.size(0), opts.num_domains)).to(opts.device)
    c_org_trans[:, 1] = 1

    # to achieve reconstruction for anomaly data use label = [1, 0]
    # to achieve reconstruction for healthy data use label = [0, 1]
    c_org_recon = torch.zeros((anomaly_images.size(0), opts.num_domains)).to(opts.device)
    c_org_recon[:, 0] = 1

    images = anomaly_images.to(opts.device).detach()
    c_org_trans = c_org_trans.to(opts.device).detach()
    c_org_recon = c_org_recon.to(opts.device).detach()

    with torch.no_grad():
        # for group forward transfer you will need 1 image
        # for translation c_org_trans will need to be the labels of the corresponding images
        # num = number of times to sample the attribute latent space
        output_b, diff_b_pos, diff_b_neg, diff_b_pos_std, diff_b_neg_std = model.test_forward_random_group(images,
                                                                                                           c_org_trans,
                                                                                                           num=100)
        # for reconstruction c_org_recon will need to be the label of the opposite class
        output_a, diff_a_pos, diff_a_neg, diff_a_pos_std, diff_a_neg_std = model.test_forward_random_group(images,
                                                                                                           c_org_recon,
                                                                                                           num=100)

        assembled_images = torch.cat(
            (images.cpu()[0:1, ::], output_b.cpu()[0:1, ::], diff_b_pos.cpu()[0:1, ::], diff_b_pos_std.cpu()[0:1, ::],
             output_a.cpu()[0:1, ::],
             diff_a_pos.cpu()[0:1, ::], diff_a_pos_std.cpu()[0:1, ::]), 3)
        # saved image: 'input_a', 'trans_image', 'trans_diff_mean',
        # 'trans_diff_var', 'recon_image', 'recon_diff_mean', 'recon_diff_var'
        name = 'group_translation'
        img_filename = '%s/%s.jpg' % (path, name)
        # saving for 2D inputs only
        torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)

        # for forward transfer images will need to be batch size 2 - of the 2 images you want to transfer
        # c_org will need to be the labels of the corresponding images
        images = torch.cat((healthy_images, anomaly_images), dim=0).type(torch.FloatTensor)
        c_org = torch.cat((c_org_trans, c_org_recon), dim=0).type(torch.FloatTensor)
        images = images.to(opts.device).detach()
        c_org = c_org.to(opts.device).detach()

        outputs_a, diff_map_a_pos, diff_map_a_neg, outputs_b, diff_map_b_pos, diff_map_b_neg = model.test_forward_transfer(
            images, c_org)

        assembled_images_1 = torch.cat(
            (healthy_images.cpu()[0:1, ::], outputs_a.cpu()[0:1, ::], diff_map_a_pos.cpu()[0:1, ::]), 3)
        assembled_images_2 = torch.cat(
            (anomaly_images.cpu()[0:1, ::], outputs_b.cpu()[0:1, ::],
             diff_map_b_pos.cpu()[0:1, ::]), 3)
        assembled_images = torch.cat((assembled_images_1, assembled_images_2), 2)

        # saved image: 'input_a', 'trans_a_to_b_image', 'trans_a_to_b_diff',
        # & 'input_b', 'trans_b_to_a_image', 'trans_b_to_a_diff'
        name = 'forward_translation'
        img_filename = '%s/%s.jpg' % (path, name)
        # saving for 2D inputs only
        torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)

        
def _save_best_models(opts, model):
    """
    Save all models based on validation results
    :param opts:
    :param model:
    :return:
    """
    if val_f1[ep] >= np.max(val_f1):
        saver.write_model(ep, total_it, iter_counter, model, model_name='model_f1')
    if val_accuracy[ep] >= np.max(val_accuracy):
        saver.write_model(ep, total_it, iter_counter, model, model_name='model_accuracy')
    if opts.cross_corr and (val_cross_corr_b[ep] >= np.max(val_cross_corr_b)):
        saver.write_model(ep, total_it, iter_counter, model, model_name='model_cc_b')
    if opts.cross_corr and (val_cross_corr_a[ep] >= np.max(val_cross_corr_a)):
        saver.write_model(ep, total_it, iter_counter, model, model_name='model_cc_a')
    if opts.regression and (val_mae[ep] <= np.min(val_mae[:ep+1])):
        saver.write_model(ep, total_it, iter_counter, model, model_name='model_mae')


def _plot_results(opts, e):
    """
    Plot all results
    :param opts:
    :param e: array of epochs
    :return:
    """
    plt.figure()
    plt.plot(e[:ep + 1], val_f1[:ep + 1], label='val_f1 score')
    plt.xlabel('val save iterations')
    plt.legend()
    plt.title('val_f1 score')
    plt.savefig(opts.results_path + '/val_f1.png')
    plt.close()

    plt.figure()
    plt.plot(e[:ep + 1], val_accuracy[:ep + 1], label='val_accuracy score')
    plt.xlabel('val save iterations')
    plt.legend()
    plt.title('val_accuracy score')
    plt.savefig(opts.results_path + '/val_accuracy.png')
    plt.close()

    plt.figure()
    plt.plot(e[:ep + 1], val_precision[:ep + 1], label='val_precision score')
    plt.xlabel('val save iterations')
    plt.legend()
    plt.title('val_precision score')
    plt.savefig(opts.results_path + '/val_precision.png')
    plt.close()

    plt.figure()
    plt.plot(e[:ep + 1], val_recall[:ep + 1], label='val_recall score')
    plt.xlabel('val save iterations')
    plt.legend()
    plt.title('val_recall score')
    plt.savefig(opts.results_path + '/val_recall.png')
    plt.close()

    if opts.regression:
        np.save(opts.results_path + '/val_mse.npy', val_mse)
        np.save(opts.results_path + '/val_mae.npy', val_mae)

        plt.figure()
        plt.plot(e[:ep + 1], val_mse[:ep + 1], label='val_mse score')
        plt.xlabel('val save iterations')
        plt.legend()
        plt.title('val_mse score')
        plt.savefig(opts.results_path + '/val_mse.png')
        plt.close()

        plt.figure()
        plt.plot(e[:ep + 1], val_mae[:ep + 1], label='val_mae score')
        plt.xlabel('val save iterations')
        plt.legend()
        plt.title('val_mae score')
        plt.savefig(opts.results_path + '/val_mae.png')
        plt.close()

    if opts.cross_corr:
        plt.figure()
        plt.plot(e[:ep + 1], val_cross_corr_a[:ep + 1], label='val_cross_corr score')
        plt.xlabel('val save iterations')
        plt.legend()
        plt.title('val_cross_corr score')
        plt.savefig(opts.results_path + '/val_cross_corr_a.png')
        plt.close()

        plt.figure()
        plt.plot(e[:ep + 1], val_cross_corr_b[:ep + 1], label='val_cross_corr score')
        plt.xlabel('val save iterations')
        plt.legend()
        plt.title('val_cross_corr score')
        plt.savefig(opts.results_path + '/val_cross_corr_b.png')
        plt.close()


def _plot_results_crossval(opts, e, fold):
    """
    Plot all results
    :param opts:
    :param e: array of epochs
    :return:
    """
    plt.figure()
    plt.plot(e[:ep + 1], val_f1[:ep + 1], label='val_f1 score')
    plt.xlabel('val save iterations')
    plt.legend()
    plt.title('val_f1 score')
    plt.savefig(opts.results_path + '/val_f1_fold' + str(fold) + '.png')
    plt.close()

    plt.figure()
    plt.plot(e[:ep + 1], val_accuracy[:ep + 1], label='val_accuracy score')
    plt.xlabel('val save iterations')
    plt.legend()
    plt.title('val_accuracy score')
    plt.savefig(opts.results_path + '/val_accuracy_fold' + str(fold) + '.png')
    plt.close()

    plt.figure()
    plt.plot(e[:ep + 1], val_precision[:ep + 1], label='val_precision score')
    plt.xlabel('val save iterations')
    plt.legend()
    plt.title('val_precision score')
    plt.savefig(opts.results_path + '/val_precision_fold' + str(fold) + '.png')
    plt.close()

    plt.figure()
    plt.plot(e[:ep + 1], val_recall[:ep + 1], label='val_recall score')
    plt.xlabel('val save iterations')
    plt.legend()
    plt.title('val_recall score')
    plt.savefig(opts.results_path + '/val_recall_fold' + str(fold) + '.png')
    plt.close()

    if opts.regression:
        np.save(opts.results_path + '/val_mse_fold' + str(fold) + '.npy', val_mse)
        np.save(opts.results_path + '/val_mae_fold' + str(fold) + '.npy', val_mae)

        plt.figure()
        plt.plot(e[:ep + 1], val_mse[:ep + 1], label='val_mse score')
        plt.xlabel('val save iterations')
        plt.legend()
        plt.title('val_mse score')
        plt.savefig(opts.results_path + '/val_mse_fold' + str(fold) + '.png')
        plt.close()

        plt.figure()
        plt.plot(e[:ep + 1], val_mae[:ep + 1], label='val_mae score')
        plt.xlabel('val save iterations')
        plt.legend()
        plt.title('val_mae score')
        plt.savefig(opts.results_path + '/val_mae_fold' + str(fold) + '.png')
        plt.close()

    if opts.cross_corr:
        plt.figure()
        plt.plot(e[:ep + 1], val_cross_corr_a[:ep + 1], label='val_cross_corr score')
        plt.xlabel('val save iterations')
        plt.legend()
        plt.title('val_cross_corr score')
        plt.savefig(opts.results_path + '/val_cross_corr_a_fold' + str(fold) + '.png')
        plt.close()

        plt.figure()
        plt.plot(e[:ep + 1], val_cross_corr_b[:ep + 1], label='val_cross_corr score')
        plt.xlabel('val save iterations')
        plt.legend()
        plt.title('val_cross_corr score')
        plt.savefig(opts.results_path + '/val_cross_corr_b_fold' + str(fold) + '.png')
        plt.close()


if __name__ == '__main__':
    main()
