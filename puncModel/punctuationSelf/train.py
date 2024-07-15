import argparse
import functools
import os
import time
from datetime import timedelta
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
from visualdl import LogWriter

from utils.reader import PuncDatasetFromErnieTokenizer, collate_fn
from utils.model import ErnieLinear
from utils.utils import add_arguments, print_arguments
from utils.logger import setup_logger

logger = setup_logger(__name__)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size', int, 128, '训练的批量大小')
add_arg('max_seq_len', int, 240, '训练数据的最大长度')
add_arg('num_workers', int, 12, '读取数据的线程数量')
add_arg('num_epoch', int, 40, '训练的轮数')
add_arg('learning_rate', float, 2.0e-5, '初始学习率的大小')
add_arg('train_data_path', str, 'dataset/train.txt', '训练数据的数据文件路径')
add_arg('dev_data_path', str, 'dataset/dev.txt', '测试数据的数据文件路径')
add_arg('punc_path', str, 'dataset/punc_vocab', '标点符号字典路径')
add_arg('model_path', str, 'models/', '保存检查点的目录')
add_arg('resume_model', str, None, '恢复训练模型文件夹')
add_arg('pretrained_token', str, r'D:\work\asrapp\puncModel\punctuationSelf\ernie-3.0-nano-zh',
        '使用的ERNIE模型权重，具体查看：https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers/ERNIE/contents.html#ernie')
args = parser.parse_args()
print_arguments(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train():
    writer = LogWriter(logdir='log')

    train_dataset = PuncDatasetFromErnieTokenizer(data_path=args.train_data_path,
                                                  punc_path=args.punc_path,
                                                  pretrained_token=args.pretrained_token,
                                                  max_seq_len=args.max_seq_len)
    dev_dataset = PuncDatasetFromErnieTokenizer(data_path=args.dev_data_path,
                                                punc_path=args.punc_path,
                                                pretrained_token=args.pretrained_token,
                                                max_seq_len=args.max_seq_len)

    # train_batch_sampler = CustomBatchSampler(train_dataset,
    #                                          batch_size=args.batch_size,
    #                                          drop_last=True,
    #                                          shuffle=True)
    train_loader = DataLoader(train_dataset,
                              collate_fn=collate_fn,
                              # batch_sampler=train_batch_sampler,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=False)

    dev_loader = DataLoader(dev_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=collate_fn,
                            drop_last=False)
    logger.info('预处理数据集完成！')
    # num_classes为字符分类大小
    model = ErnieLinear(pretrained_token=args.pretrained_token, num_classes=len(train_dataset.punc2id),
                        from_hf_hub=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=1.0e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epoch)

    # 恢复训练
    last_epoch = 0
    # if args.resume_model:
    #     model.load_state_dict(torch.load(os.path.join(args.resume_model, 'model.pt')))
    #     optimizer.load_state_dict(torch.load(os.path.join(args.resume_model, 'optimizer.pt')))

    best_loss = 1e3
    train_step, test_step = 0, 0
    train_times = []
    sum_batch = len(train_loader) * args.num_epoch

    for epoch in range(last_epoch, args.num_epoch):
        epoch += 1
        start = time.time()
        for batch_id, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = torch.reshape(labels, shape=[-1])
            y, logit = model(inputs)
            pred = torch.argmax(logit, dim=1)
            loss = criterion(y, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            F1_score = f1_score(labels.cpu().numpy().tolist(), pred.cpu().numpy().tolist(), average="macro")
            train_times.append((time.time() - start) * 1000)
            # 多卡训练只使用一个进程打印
            if batch_id % 100 == 0:
                eta_sec = (sum(train_times) / len(train_times)) * (
                        sum_batch - (epoch - 1) * len(train_loader) - batch_id)
                eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                logger.info(
                    'Train epoch: [{}/{}], batch: [{}/{}], loss: {:.5f}, f1_score: {:.5f}, learning rate: {:>.8f}, '
                    'eta: {}'.format(
                        epoch, args.num_epoch, batch_id, len(train_loader), loss.item(), F1_score, scheduler.get_lr()[0],
                        eta_str))

                writer.add_scalar('Train/Loss', loss.item(), train_step)
                writer.add_scalar('Train/F1_Score', F1_score, train_step)
                train_step += 1
            start = time.time()

        writer.add_scalar('Train/LearnRate', scheduler.get_lr()[0], epoch)
        scheduler.step()
        model.eval()
        eval_loss = []
        eval_f1_score = []
        for batch_id, (inputs, labels) in enumerate(dev_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = torch.reshape(labels, shape=[-1])
            y, logit = model(inputs)
            pred = torch.argmax(logit, dim=1)
            loss = criterion(y, labels)
            eval_loss.append(loss.item())
            F1_score = f1_score(labels.cpu().numpy().tolist(), pred.cpu().numpy().tolist(), average="macro")
            eval_f1_score.append(F1_score)
            if batch_id % 100 == 0:
                logger.info('Batch: [{}/{}], loss: {:.5f}, f1_score: {:.5f}'.format(
                    batch_id, len(dev_loader), loss.item(), F1_score))
        eval_loss1 = sum(eval_loss) / len(eval_loss)
        eval_f1_score1 = sum(eval_f1_score) / len(eval_f1_score)
        if eval_loss1 < best_loss:
            best_loss = eval_loss1
            # 保存最优模型

            save_dir = os.path.join(args.model_path, "best_checkpoint")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
            torch.save(optimizer.state_dict(), os.path.join(save_dir, 'optimizer.pt'))
            logger.info(f'模型保存在：{save_dir}')
        logger.info('Avg eval, loss: {:.5f}, f1_score: {:.5f} best loss: {:.5f}'.
                    format(eval_loss1, eval_f1_score1, best_loss))
        model.train()

        writer.add_scalar('Test/Loss', eval_loss1, test_step)
        writer.add_scalar('Test/F1_Score', eval_f1_score1, test_step)
        save_dir = os.path.join(args.model_path, "checkpoint")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(save_dir, 'optimizer.pt'))
        logger.info(f'模型保存在：{save_dir}')
        test_step += 1


if __name__ == "__main__":
    train()
