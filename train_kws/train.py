import argparse
import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss import ArcMarginLoss, MarginScheduler
from model import CAMPPlusClassifier
from speakerlab.utils.config import build_config
from speakerlab.utils.epoch import EpochLogger
from speakerlab.utils.utils import set_seed, get_logger, AverageMeters, ProgressMeter, accuracy
from utils import SRDataset

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
    train_dataset = SRDataset()
    train_loader = DataLoader(train_dataset, batch_size=int(config.batch_size), shuffle=True)

    # model
    m = torch.load("campplus_cn_common.bin")
    model = CAMPPlusClassifier(train_dataset.now_label)
    model.encoder.load_state_dict(m)
    model.cuda()

    # optimizer
    optimizer = torch.optim.Adam([{"params": model.encoder.parameters(), "lr": 5e-6},
                                  {"params": model.classifier.parameters(), "lr": 1e-4}], lr=config.lr)

    # loss function
    criterion = torch.nn.CrossEntropyLoss()
    criterion.cuda()

    # scheduler

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # margin_scheduler = MarginScheduler(criterion, 0.0, 0.2, 15, 25)

    # others
    epoch_logger = EpochLogger(save_file=os.path.join(config.exp_dir, 'train_epoch.log'))

    cudnn.benchmark = True

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

        epoch_logger.log_stats(
            stats_meta={"epoch": epoch},
            stats=train_stats,
        )
        # save checkpoint
        if epoch % config.save_epoch_freq == 0:
            print("save model at epoch %d" % epoch)
            torch.save(model, os.path.join(config.exp_dir, f'model_{epoch}.pth'))


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
    for i, (x, y) in enumerate(train_loader):
        # data loading time
        train_stats.update('Data', time.time() - end)

        x = x.cuda()
        y = y.cuda()

        # compute output
        output = model(x)
        loss = criterion(output, y)
        acc1 = accuracy(output, y)

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
    key_stats = {
        'Avg_loss': train_stats.avg('Loss'),
        'Avg_acc': train_stats.avg('Acc@1'),
        'Lr_value': train_stats.val('Lr')
    }
    return key_stats


if __name__ == '__main__':
    main()
