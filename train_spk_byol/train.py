import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from byol_pytorch import BYOL
from model import CAMPPlus
from utils import get_datasets

BATCH_SIZE = 48
EPOCHS = 40
LR = 2e-5
NUM_GPUS = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(learner, train_loader):
    loss_sum = 0
    model.train()
    for x1, x2 in tqdm(train_loader):
        x1 = x1.to(device)
        x2 = x2.to(device)
        loss = learner(x1, x2)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        learner.update_moving_average()
        loss_sum += loss.item()

    return loss_sum / len(train_loader)


def validate(learner, val_loader):
    loss_sum = 0
    model.eval()
    with torch.no_grad():
        for i, x1, x2 in enumerate(val_loader):
            loss = learner(x1, x2)
            loss_sum += loss.item()

    return loss_sum / len(val_loader)


if __name__ == '__main__':
    model_params = torch.load("../train_kws/campplus_cn_common.bin")
    model = CAMPPlus()
    model.load_state_dict(model_params)
    model = model.to(device)
    train_dataset, valid_dataset = get_datasets()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    learner = BYOL(model, hidden_layer=-1, projection_hidden_size=192)

    for epoch in tqdm(range(EPOCHS)):
        train_loss = train(learner, train_loader)
        # valid_loss = validate(learner, valid_loader)
        print(f"epoch {epoch}: train loss {train_loss}")
        if (epoch+1) % 5 == 0:
            torch.save(model, f"checkpoints/model_{epoch+1}.pt")
    torch.save(model, "checkpoints/model_final.pt")
