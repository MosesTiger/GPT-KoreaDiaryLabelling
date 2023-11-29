import random
import argparse
import datetime

import numpy as np
from matplotlib import pyplot as plt
import torch

from utils.dataloader import get_dataloader
from utils.model import WeekChallenge

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser(description='nlp final project')
parser.add_argument('--epoch', default=20, type=int, help='epochs')

parser.add_argument('--batch-size', default=16, type=int, help='batch size')
parser.add_argument('--optim', default='Adam', type=str, help='SGD, Adam, AdamW')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--model-size', default='base', type=str, help='base, small, large')

parser.add_argument('--data-path', default='data/gpt_labeled_diary.json', type=str, help='data path')
parser.add_argument('--model-save-path', default='ckpt', type=str, help='model save path')
parser.add_argument('--loss-save-path', default='loss_history', type=str, help='loss save path')
parser.add_argument('--log-interval', default=10, type=int, help='log interval')
parser.add_argument('--set-seed', default=True, type=bool, help='set seed')
parser.add_argument('--seed', default=0, type=int, help='seed')

parser.add_argument('--use-cuda', default=True, type=bool, help='use cuda')

args = parser.parse_args()

if __name__ == "__main__":
    
    if args.set_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    print("get dataloader")
    train_dataloader, val_dataloader = get_dataloader(args.data_path, args.batch_size)

    print("train data size: {}".format(len(train_dataloader.dataset)))
    print("val data size: {}".format(len(val_dataloader.dataset)))

    print("get model", args.model_size)
    model_args = {
        "model_size": args.model_size,
        "use_cuda": args.use_cuda
    }

    mynet = WeekChallenge(model_args)

    print("get optimizer", args.optim)
    if args.optim == "SGD":
        optimizer = torch.optim.SGD(mynet.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=1e-4)
    elif args.optim == "Adam":
        optimizer = torch.optim.Adam(mynet.parameters(), 
                                     lr=args.lr,
                                     weight_decay=1e-4,
                                     betas=(0.9, 0.999),
                                     eps=1e-8,)
    elif args.optim == "AdamW":
        optimizer = torch.optim.AdamW(mynet.parameters(),
                                      lr=args.lr,
                                      weight_decay=1e-4,
                                      betas=(0.9, 0.999),
                                      eps=1e-8,)
    
    criterion = torch.nn.MSELoss()

    train_loss_history = []
    val_loss_history = []

    for i in range(args.epoch):
        print("epoch: {}".format(i+1))
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        mynet.train()
        diff_sum = torch.zeros((5))
        for idx, (content, labels) in enumerate(train_dataloader):
            if args.use_cuda:
                labels = labels.cuda()
                diff_sum = diff_sum.cuda()
            optimizer.zero_grad()
            outputs = mynet(content)
            loss = criterion(outputs, labels)
            loss = torch.sqrt(loss)
            loss.backward()
            optimizer.step()

            diff_sum += torch.sum(torch.abs(outputs-labels), dim=0)
            if idx % args.log_interval == 0:
                if idx != 0:
                    processed = idx*args.batch_size
                    progress = processed/len(train_dataloader.dataset)*100
                    print(f"{progress:>.2f}%, train loss: {loss.item():>.5f}")

        print(f"train mae: {diff_sum/len(train_dataloader.dataset)}")
        train_loss_history.append(loss.item())

        mynet.eval()
        with torch.no_grad():
            diff_sum = torch.zeros((5))
            if args.use_cuda:
                diff_sum = diff_sum.cuda()
            sum = 0
            sum_loss = 0.
            for idx, (content, labels) in enumerate(val_dataloader):
                if args.use_cuda:
                    labels = labels.cuda()
                outputs = mynet(content)
                loss = criterion(outputs, labels)
                loss = torch.sqrt(loss)
                sum_loss += loss.item()
                sum += 1
                diff_sum += torch.sum(torch.abs(outputs-labels), dim=0)
            print(f"val loss: {sum_loss/sum:>.5f}")
            print(f"val mae: {diff_sum/len(val_dataloader.dataset)}")
            val_loss_history.append(loss.item())

    save_name = "{}/{}_{}_{}_{}.pth".format(args.model_save_path, args.batch_size, args.optim, args.lr, args.model_size)
    torch.save(mynet.state_dict(), save_name)
    print("model saved to {}".format(save_name))

    # draw loss curve
    plt.plot(train_loss_history, label='train')
    plt.plot(val_loss_history, label='val')
    plt.legend()
    plt.savefig("{}/{}_{}_{}_{}.png".format(args.loss_save_path, args.batch_size, args.optim, args.lr, args.model_size))