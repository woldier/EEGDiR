"""
Author: wolider wong
Date: 2024-1-11
Description:
"""
import datetime, pytz, os, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import AbstractDenoiser
from utils.data_show import show_img  # visualization tool
# evaluate tool
from utils.evalueate_func import RRMSE_spectral, RRMSE_temporal, CC, compute_params, compute_storage_size
from utils.train_valid_utils import get_config, init_model, check_dir, config_backpack, load_dataset, init_optimizer
from accelerate import Accelerator

# =========================config path===========================================
config_path = {
    r'config/retnet/config.yml',
    r'../config/EEGDnet/config.yml',
    r'../config/SCNN/config.yml',
    r'../config/1DResCNN/config.yml'
}
config_path = config_path[0]  # choose which path you want to use
# =====================================================================
# Get the desired time zone object
# For example, the Shanghai time zone is used here
tz = pytz.timezone('Asia/Shanghai')
date_str = datetime.datetime.now(tz).strftime("%Y_%m_%d_%H")


def train_loop(net: AbstractDenoiser, train_set, test_set, optimizer: torch.optim.Optimizer, criterion, config: dict,
               base_dir):
    print("===========================woldier Deep Learning Distribution Framework====================================")
    best_test_loss = 1000
    data_loader_train = DataLoader(train_set, shuffle=True, batch_size=config["train"]["batch_size"])
    data_loader_test = DataLoader(test_set, shuffle=False, batch_size=config["train"]["batch_size"])
    epochs = config["train"]["epochs"]
    # ==================================Accelerator Distribution Training===========================================
    accelerator = Accelerator()
    device = accelerator.device
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    net, optimizer, data_loader_train, data_loader_test, scheduler = accelerator.prepare(
        net, optimizer, data_loader_train, data_loader_test, scheduler
    )
    # =============================================================================================================
    for epoch in range(epochs):
        start = time.time()
        # initialize  loss value for every epoch
        train_loss, test_loss = 0, 0
        # =============================train====================================
        train_loss = train_loop_one(
            data_loader_train,
            epoch,
            net,
            optimizer,
            train_loss,
            accelerator
        )
        # =============================test====================================
        rrmse_t, rrmse_s, cc = .0, .0, .0
        cc, rrmse_s, rrmse_t, test_loss = valid_loop_one(base_dir,
                                                         cc,
                                                         data_loader_test,
                                                         epoch,
                                                         net,
                                                         rrmse_s,
                                                         rrmse_t,
                                                         test_loss,
                                                         accelerator,
                                                         (epoch + 1) % config["train"]["save_img_rate"] == 0)
        # =============================logging====================================
        best_test_loss = logging(base_dir,
                                 best_test_loss,
                                 cc, epoch,
                                 epochs,
                                 net,
                                 rrmse_s,
                                 rrmse_t,
                                 start,
                                 test_loss,
                                 train_loss,
                                 accelerator
                                 )


def train_loop_one(data_loader_train, epoch, net, optimizer, train_loss, accelerator):
    with tqdm(total=len(data_loader_train), position=0, leave=True) as pbar:
        net.train()
        for step, batch in enumerate(data_loader_train):
            inputs = batch["y"]
            labels = batch["x"]
            outputs, m_loss = net(inputs, labels)
            # m_loss = criterion(outputs, labels)
            train_loss += m_loss.item()
            optimizer.zero_grad()
            accelerator.backward(m_loss)  # backward
            # m_loss.backward()  # backward
            optimizer.step()  # optimizer
            pbar.update()
            pbar.set_description("epoch%d :Train Loss %f" % (epoch, train_loss / (step + 1)))  # 设置描述
    train_loss = train_loss / float(len(data_loader_train))
    return train_loss


def valid_loop_one(base_path, cc, data_loader_test, epoch, net, rrmse_s, rrmse_t, test_loss, accelerator,
                   save_img: bool = False):
    index = np.random.randint(0, int(len(data_loader_test)), dtype="int")
    with tqdm(total=int(len(data_loader_test)), position=0, leave=True) as pbar, torch.no_grad():
        net.eval()
        for step, batch in enumerate(data_loader_test):
            inputs = batch["y"]
            labels = batch["x"]
            std = batch["std"]
            outputs, m_loss = net(inputs, labels)
            outputs, labels, std = accelerator.gather_for_metrics((outputs, labels, std))
            if index == step and save_img:
                show_img(labels.cpu().detach().numpy() * std.cpu().detach().numpy(),
                         inputs.cpu().detach().numpy() * std.cpu().detach().numpy(),
                         outputs.cpu().detach().numpy() * std.cpu().detach().numpy(),
                         save_name=os.path.join(base_path, "img", "{}.svg".format(epoch)),
                         max_batch_size=16,
                         format='svg',
                         dpi=50
                         )
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


def logging(base_path, best_test_loss, cc, epoch, epochs, net, rrmse_s, rrmse_t, start, test_loss, train_loss,
            accelerator):
    if best_test_loss > test_loss:
        accelerator.wait_for_everyone()
        torch.save(accelerator.unwrap_model(net).state_dict(),
                   os.path.join(base_path, "weight", "best.pth"))
        best_test_loss = test_loss
    if epoch % 1 == 0:
        accelerator.wait_for_everyone()
        torch.save(accelerator.unwrap_model(net).state_dict(),
                   os.path.join(base_path, "weight", f"EPOCH{epoch}.pth"))
    log_str = '''\nEpoch #: {}/{}, Time taken: {} secs,\nLosses: train_MSE= {},test_MSE={} \n rrmse_t= {}, rrmse_s= {}, cc={}'''.format(
        epoch + 1, epochs, time.time() - start, train_loss, test_loss, rrmse_t, rrmse_s, cc)
    print(log_str)
    f = open(os.path.join(base_path, "logs", "log.txt"), "a")
    f.writelines(log_str)
    f.close()  # close file
    return best_test_loss


def run():
    print(f"loading {config_path}")
    config = get_config(config_path)
    # check dir
    base_dir = "./results/{}/".format(date_str + "_" + config["logs"]["name"])
    check_dir(base_dir)
    # save config backpack
    config_backpack(config_path, base_dir)
    # init model
    model = init_model(config)
    train_dataset, test_dataset = load_dataset(config)
    # Configuration Optimizer
    optim = init_optimizer(model, config)
    criterion = torch.nn.MSELoss()
    train_loop(model, train_dataset, test_dataset, optim, criterion, config, base_dir)


if __name__ == "__main__":
    run()
