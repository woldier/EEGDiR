"""
Author: wolider wong
Date: 2024-1-11
Description:
"""
import datetime
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data_show import show_img  # visualization tool
# evaluate tool
from utils.evalueate_func import RRMSE_spectral, RRMSE_temporal, CC, compute_params, compute_storage_size
from utils.train_valid_utils import get_config, init_model, check_dir, config_backpack, load_dataset

config_path = r'../config/retnet/config.yml'
# config_path = r'../config/EEGDnet/config.yml'
# config_path = r'../config/LSTM/config.yml'
# config_path = r'../config/SCNN/config.yml'
# config_path = r'../config/1DResCNN/config.yml'
date_str = datetime.datetime.now().strftime("%Y_%m_%d_%H")


def train_loop(net: torch.nn.Module, train_set, test_set, optimizer: torch.optim.Optimizer, criterion, config: dict):
    best_test_loss = 1000
    data_loader_train = DataLoader(train_set, shuffle=True, batch_size=config["train"]["batch_size"])
    data_loader_test = DataLoader(test_set, shuffle=False, batch_size=config["train"]["batch_size"])
    net.cuda()
    epochs = config["train"]["epochs"]
    base_path = date_str + "_" + config["logs"]["name"]  # YYYY_mm_DD_HH_XXX
    for epoch in range(epochs):
        start = time.time()
        # initialize  loss value for every epoch
        train_loss, test_loss = 0, 0
        # =============================train====================================
        train_loss = train_loop_one(criterion,
                                    data_loader_train,
                                    epoch,
                                    net,
                                    optimizer,
                                    train_loss)
        # =============================test====================================
        rrmse_t, rrmse_s, cc = .0, .0, .0
        cc, rrmse_s, rrmse_t, test_loss = valid_loop_one(base_path,
                                                         cc,
                                                         criterion,
                                                         data_loader_test,
                                                         epoch,
                                                         net,
                                                         rrmse_s,
                                                         rrmse_t,
                                                         test_loss,
                                                         (epoch + 1) % config["train"]["save_img_rate"] == 0)
        # =============================logging====================================
        logging(base_path,
                best_test_loss,
                cc, epoch,
                epochs,
                net,
                rrmse_s,
                rrmse_t,
                start,
                test_loss,
                train_loss)


def train_loop_one(criterion, data_loader_train, epoch, net, optimizer, train_loss):
    with tqdm(total=len(data_loader_train), position=0, leave=True) as pbar:
        net.train()
        for step, batch in enumerate(data_loader_train):
            inputs = batch["y"].cuda()
            labels = batch["x"].cuda()
            outputs = net(inputs)
            m_loss = criterion(outputs, labels)
            train_loss += m_loss.item()
            optimizer.zero_grad()
            m_loss.backward()  # backward
            optimizer.step()  # optimizer
            pbar.update()
            pbar.set_description("epoch%d :Train Loss %f" % (epoch, train_loss / (step + 1)))  # 设置描述
    train_loss = train_loss / float(len(data_loader_train))
    return train_loss


def valid_loop_one(base_path, cc, criterion, data_loader_test, epoch, net, rrmse_s, rrmse_t, test_loss,
                   save_img: bool = False):
    index = np.random.randint(0, int(len(data_loader_test)), dtype="int")
    with tqdm(total=int(len(data_loader_test)), position=0, leave=True) as pbar:
        net.eval()
        for step, batch in enumerate(data_loader_test):
            inputs = batch["y"].cuda()
            labels = batch["x"].cuda()
            outputs = net(inputs)
            if index == step and save_img:
                show_img(labels.cpu().detach().numpy(),
                         inputs.cpu().detach().numpy(),
                         outputs.cpu().detach().numpy(),
                         save_name="../results/{}/img/{}.svg".format(base_path, epoch),
                         max_batch_size=16,
                         format='svg',
                         dpi=50
                         )
            m_loss = criterion(outputs, labels)
            test_loss += m_loss.item()
            rrmse_t += RRMSE_temporal(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            rrmse_s += RRMSE_spectral(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            cc += CC(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            pbar.update()
            pbar.set_description("epoch%d :Valid Loss %f" % (epoch, test_loss / (step + 1)))  # 设置描述
        pbar.close()
    test_loss = test_loss / float(len(data_loader_test))
    rrmse_t = rrmse_t / float(len(data_loader_test))
    rrmse_s = rrmse_s / float(len(data_loader_test))
    cc = cc / float(len(data_loader_test))
    return cc, rrmse_s, rrmse_t, test_loss


def logging(base_path, best_test_loss, cc, epoch, epochs, net, rrmse_s, rrmse_t, start, test_loss, train_loss):
    if best_test_loss > test_loss:
        torch.save(net.state_dict(),
                   "../results/{}/weight/".format(base_path) + "best" + ".pth")
        best_test_loss = test_loss
    if epoch % 10 == 9:
        torch.save(net.state_dict(),
                   "../results/{}/weight/".format(base_path) + "EPOCH" + str(
                       epoch) + ".pth")
    log_str = '''\nEpoch #: {}/{}, Time taken: {} secs,\nLosses: train_MSE= {},test_MSE={} \n rrmse_t= {}, rrmse_s= {}, cc={}'''.format(
        epoch + 1, epochs, time.time() - start, train_loss, test_loss, rrmse_t, rrmse_s, cc)
    print(log_str)
    f = open("../results/{}/logs/log.txt".format(base_path), "a")
    f.writelines(log_str)
    f.close()  # close file


def run():
    config = get_config(config_path)
    # check dir
    base_dir = "../results/{}/".format(date_str + "_" + config["logs"]["name"])
    check_dir(base_dir)
    # save config backpack
    config_backpack(config_path, base_dir)
    # init model
    model = init_model(config)
    # compute Params
    compute_params(model)
    # compute storge size
    # load dataset
    train_dataset, test_dataset = load_dataset(config)
    # 配置优化器
    optim = torch.optim.AdamW(model.parameters(), lr=0.0005, betas=(0.5, 0.9), eps=1e-08)
    criterion = torch.nn.MSELoss()
    train_loop(model, train_dataset, test_dataset, optim, criterion, config)


if __name__ == "__main__":
    run()
