import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CAMPPlusClassifier
from speakerlab.utils.config import build_config
from speakerlab.utils.epoch import EpochLogger
from speakerlab.utils.utils import set_seed, get_logger, AverageMeters, ProgressMeter, accuracy, accuracyNum
from utils import get_datasets

parser = argparse.ArgumentParser(description='Speaker Network Training')
parser.add_argument('--config', default='cam++.yaml', type=str, help='Config file for training')
parser.add_argument('--resume', default=True, type=bool, help='Resume from recent checkpoint or not')
parser.add_argument('--seed', default=1234, type=int, help='Random seed for training.')


def main():
    args, overrides = parser.parse_known_args(sys.argv[1:])
    config = build_config(args.config, None, True)

    set_seed(args.seed)

    os.makedirs(config.exp_dir, exist_ok=True)
    logger = get_logger('%s/train.log' % config.exp_dir)

    # dataset
    # dataloader

    train_dataset, valid_dataset = get_datasets()
    train_loader = DataLoader(train_dataset, batch_size=int(config.batch_size), shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=int(config.batch_size), shuffle=True, num_workers=4)

    # model
    m = torch.load("campplus_cn_common.bin")
    model = CAMPPlusClassifier(train_dataset.now_label)
    model.encoder.load_state_dict(m)
    model.cuda()

    # optimizer
    optimizer = torch.optim.Adam([{"params": model.encoder.parameters(), "lr": 1e-6},
                                  {"params": model.classifier.parameters(), "lr": 1e-4}], lr=config.lr)

    # loss function
    criterion = torch.nn.CrossEntropyLoss()
    criterion.cuda()

    # scheduler

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # margin_scheduler = MarginScheduler(criterion, 0.0, 0.2, 15, 25)

    # others
    epoch_logger = EpochLogger(save_file=os.path.join(config.exp_dir, 'train_epoch.log'))

    cudnn.benchmark = True
    metric = {"train_acc": [], "train_loss": [], "valid_acc": [], "valid_loss": []}
    for epoch in tqdm(range(int(config.num_epoch))):

        # train one epoch
        train_stats = train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            lr_scheduler,
            logger,
            config,
        )

        test_stats = validate(valid_loader, model, criterion, epoch, logger, config)
        metric["train_acc"].append(train_stats["Avg_acc"])
        metric["train_loss"].append(train_stats["Avg_loss"])
        metric["valid_acc"].append(test_stats["Avg_acc"])
        metric["valid_loss"].append(test_stats["Avg_loss"])
        np.save(os.path.join(config.exp_dir, "metric.npy"), np.array(metric))
        epoch_logger.log_stats(
            stats_meta={"epoch": epoch},
            stats=train_stats,
        )
        # save checkpoint
        if epoch % config.save_epoch_freq == 0:
            print("save model at epoch %d" % epoch)
            torch.save(model, os.path.join(config.exp_dir, f'model_{epoch}.pth'))
    # np.save("metric.npy", np.array(metric))
    torch.save(model, os.path.join(config.exp_dir, f'model_final.pth'))


def train(train_loader, model, criterion, optimizer, epoch, lr_scheduler, logger, config):
    train_stats = AverageMeters()
    train_stats.add('Time', ':6.3f')
    train_stats.add('Data', ':6.3f')
    train_stats.add('Loss', ':.4e')
    train_stats.add('Acc@1', ':6.2f')
    train_stats.add('Lr', ':.3e')
    train_stats.add('Margin', ':.3f')
    progress = ProgressMeter(
        len(train_loader),
        train_stats,
        prefix="Epoch: [{}]".format(epoch)
    )

    # train mode
    model.train()

    end = time.time()
    loss_sum = 0
    acc_sum = 0
    total_num = 0
    for i, (x, y) in enumerate(train_loader):
        # data loading time
        train_stats.update('Data', time.time() - end)

        x = x.cuda()
        y = y.cuda()
        total_num+=y.shape[0]
        # compute output
        output = model(x)
        loss = criterion(output, y)
        acc1 = accuracyNum(output, y)
        loss_sum += loss.item()
        acc_sum += acc1.item()
        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # recording
        train_stats.update('Loss', loss.item(), x.size(0))
        train_stats.update('Acc@1', acc1.item(), x.size(0))
        train_stats.update('Lr', optimizer.param_groups[0]["lr"])
        train_stats.update('Time', time.time() - end)

        if i % config.log_batch_freq == 0:
            logger.info(progress.display(i))

        end = time.time()

    lr_scheduler.step()
    avg_loss = loss_sum / len(train_loader)
    avg_acc = acc_sum / total_num
    print(f"=============train epoch{epoch}:  avg loss: {avg_loss}, avg acc: {avg_acc}=================")
    key_stats = {
        'Avg_loss': avg_loss,
        'Avg_acc': avg_acc,
        'Lr_value': train_stats.val('Lr')
    }
    return key_stats


def validate(valid_loader, model: torch.nn.Module, criterion, epoch, logger, config):
    valid_stats = AverageMeters()
    valid_stats.add('Time', ':6.3f')
    valid_stats.add('Data', ':6.3f')
    valid_stats.add('Loss', ':.4e')
    valid_stats.add('Acc@1', ':6.2f')

    progress = ProgressMeter(
        len(valid_loader),
        valid_stats,
        prefix="Epoch: [{}]".format(epoch)
    )

    # train mode
    model.eval()

    end = time.time()
    loss_sum = 0
    acc_sum = 0
    total_num = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(valid_loader):
            # data loading time
            valid_stats.update('Data', time.time() - end)

            x = x.cuda()
            y = y.cuda()
            total_num+=y.shape[0]
            # compute output
            output = model(x)
            loss = criterion(output, y)
            acc1 = accuracyNum(output, y)
            loss_sum += loss.item()
            acc_sum += acc1.item()
            # recording
            valid_stats.update('Loss', loss.item(), x.size(0))
            valid_stats.update('Acc@1', acc1.item(), x.size(0))
            valid_stats.update('Time', time.time() - end)

            if i % config.log_batch_freq == 0:
                logger.info(progress.display(i))

            end = time.time()
    avg_loss = loss_sum / len(valid_loader)
    avg_acc = acc_sum / total_num
    print(f"=============valid epoch{epoch}:  avg loss: {avg_loss}, avg acc: {avg_acc}=================")
    key_stats = {
        'Avg_loss': avg_loss,
        'Avg_acc': avg_acc,
        # 'Lr_value': valid_stats.val('Lr')
    }
    return key_stats


if __name__ == '__main__':
    main()
