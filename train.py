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
import json
import torch
random_seed = 8


def main():
    global val_accuracy, val_f1, val_precision, val_recall, val_cross_corr_a, val_cross_corr_b, val_mse, val_mae, \
        saver, ep, save_opts, total_it, iter_counter, t0

    # initialise params
    parser = TrainOptions()
    opts = parser.parse()
    opts.random_seed = random_seed
    opts.device = opts.device if torch.cuda.is_available() and opts.gpu else 'cpu'
    opts.name = opts.data_type + '_' + time.strftime("%d%m%Y-%H%M")
    opts.results_path = os.path.join(opts.result_dir, opts.name)
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
        opts.nz = 640
    else:
        from saver import Saver
        opts.nz = 64

    print('\n--- load dataset ---')
    # add new dataloader in _load_dataloader(), and in dataloader_utils.py
    healthy_dataloader, healthy_val_dataloader, _, anomaly_dataloader, anomaly_val_dataloader, _ = _load_dataloader(opts)

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

        _validation(opts, model, healthy_val_dataloader, anomaly_val_dataloader)

    # save last model
    saver.write_model(ep, total_it, iter_counter, model, model_name='model_last')
    saver.write_img(ep, total_it, model)

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
            opts, anomaly=False, mode='train', batch_size=opts.batch_size//2)
        anomaly_dataloader = init_synth_dataloader(
            opts, anomaly=True, mode='train', batch_size=opts.batch_size//2)
        healthy_val_dataloader = init_synth_dataloader(
            opts, anomaly=False, mode='val', batch_size=opts.val_batch_size//2)
        anomaly_val_dataloader = init_synth_dataloader(
            opts, anomaly=True, mode='val', batch_size=opts.val_batch_size//2)
        healthy_test_dataloader = init_synth_dataloader(
            opts, anomaly=False, mode='test', batch_size=opts.val_batch_size//2)
        anomaly_test_dataloader = init_synth_dataloader(
            opts, anomaly=True, mode='test', batch_size=opts.val_batch_size//2)
    elif opts.data_type == 'biobank_age':
        healthy_dataloader, healthy_val_dataloader, healthy_test_dataloader, anomaly_dataloader, anomaly_val_dataloader, anomaly_test_dataloader = init_biobank_age_dataloader(opts)

    return healthy_dataloader, healthy_val_dataloader, healthy_test_dataloader, anomaly_dataloader, anomaly_val_dataloader, anomaly_test_dataloader


def _validation(opts, model, healthy_val_dataloader, anomaly_val_dataloader):
    """
    Validation function for classification and regression
    :param opts:
    :param model: networks
    :param healthy_val_dataloader:
    :param anomaly_val_dataloader:
    :return:
    """
    e = np.arange(opts.n_ep)
    val_accuracy_temp = np.zeros((0))
    val_f1_temp = np.zeros((0))
    val_precision_temp = np.zeros((0))
    val_recall_temp = np.zeros((0))
    if opts.regression:
        val_mse_temp = np.zeros((0))
        val_mae_temp = np.zeros((0))
        val_pred_temp = np.zeros((0))
        val_labels_temp = np.zeros((0))
    if opts.cross_corr:
        val_cross_corr_temp_a = np.zeros((0))
        val_cross_corr_temp_b = np.zeros((0))
    healthy_val_iter = iter(healthy_val_dataloader)
    anomaly_val_iter = iter(anomaly_val_dataloader)

    min_len = np.minimum(len(anomaly_val_dataloader), len(healthy_val_dataloader))
    for j in range(min_len):
        healthy_val_images, healthy_val_label_reg, _ = healthy_val_iter.next()
        anomaly_val_images, anomaly_val_label_reg, mask = anomaly_val_iter.next()
        healthy_val_c_org = torch.zeros((healthy_val_images.size(0), opts.num_domains)).to(opts.device)
        healthy_val_c_org[:, 0] = 1
        anomaly_val_c_org = torch.zeros((healthy_val_images.size(0), opts.num_domains)).to(opts.device)
        anomaly_val_c_org[:, 1] = 1
        images_val = torch.cat((healthy_val_images, anomaly_val_images), dim=0).type(torch.FloatTensor)
        c_org_val = torch.cat((healthy_val_c_org, anomaly_val_c_org), dim=0).type(torch.FloatTensor)
        label_reg_val = torch.cat((healthy_val_label_reg, anomaly_val_label_reg), dim=0).type(torch.FloatTensor)
        images_val = images_val.to(opts.device).detach()
        c_org_val = c_org_val.to(opts.device).detach()
        label_reg_val = label_reg_val.to(opts.device).detach()
        if len(mask.size()) > 2:
            mask = mask.to(opts.device).detach()
        else:
            mask = None

        accuracy, f1, precision, recall = model.classification_scores(images_val, c_org_val)
        val_accuracy_temp = np.append(val_accuracy_temp, accuracy)
        val_f1_temp = np.append(val_f1_temp, f1)
        val_precision_temp = np.append(val_precision_temp, precision)
        val_recall_temp = np.append(val_recall_temp, recall)
        if opts.regression:
            mse, mae, pred = model.regression(images_val, label_reg_val)
            val_mse_temp = np.append(val_mse_temp, mse)
            val_mae_temp = np.append(val_mae_temp, mae)
            val_pred_temp = np.append(val_pred_temp, pred)
            val_labels_temp = np.append(val_labels_temp,
                                        label_reg_val.squeeze(1).detach().cpu().numpy().astype(float))

        if opts.cross_corr:
            cross_corr_a, cross_corr_b = model.cross_correlation(images_val, mask, c_org_val)
            val_cross_corr_temp_a = np.append(val_cross_corr_temp_a, cross_corr_a)
            val_cross_corr_temp_b = np.append(val_cross_corr_temp_b, cross_corr_b)

    time_elapsed = time.time() - t0
    hours, rem = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)

    val_accuracy[ep] = np.mean(val_accuracy_temp)
    val_f1[ep] = np.mean(val_f1_temp)
    val_precision[ep] = np.mean(val_precision_temp)
    val_recall[ep] = np.mean(val_recall_temp)

    print('Total it: {:d} (ep {:d}, it {:d}), Val Accuracy: {:.2f}, '
          'Val F1 score: {:.2f}, Elapsed time: {:0>2}:{:0>2}:{:05.2f}'
          .format(total_it, ep, iter_counter, val_accuracy[ep], val_f1[ep], int(hours), int(minutes), seconds))

    if val_f1[ep] >= np.max(val_f1):
        save_opts['val_accuracy'] = np.max(val_accuracy[ep])
        save_opts['val_f1'] = np.max(val_f1[ep])
        save_opts['val_precision'] = np.max(val_precision[ep])
        save_opts['val_recall'] = np.max(val_recall[ep])

    if opts.regression:
        val_mse[ep] = np.mean(val_mse_temp)
        val_mae[ep] = np.mean(val_mae_temp)
        if val_mae[ep] <= np.min(val_mae[:ep]):
            save_opts['val_mse'] = val_mse[ep]
            save_opts['val_mae'] = val_mae[ep]

        print('Total it: {:d} (ep {:d}, it {:d}), Val MAE: {:.2f}, '
              'Val MSE: {:.2f}, Elapsed time: {:0>2}:{:0>2}:{:05.2f}'
              .format(total_it, ep, iter_counter, val_mae[ep], val_mse[ep], int(hours), int(minutes), seconds))

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
    if opts.regression and (val_mae[ep] <= np.min(val_mae[:ep])):
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


if __name__ == '__main__':
    main()
