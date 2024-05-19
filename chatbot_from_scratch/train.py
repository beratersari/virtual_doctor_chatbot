import argparse

from timeit import default_timer as timer

import torch

from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.cuda.amp import GradScaler
from data_utils import ConversationDataset, read_dataset

from model import Transformer
from config import PAD_IDX


def train_epoch(train_dataloader, transformer, optimizer, loss_fn, device, scaler):
    losses = 0

    transformer.train()
    for src, src_mask, tgt_input, tgt_mask, tgt_target in train_dataloader:
        src, src_mask, tgt_input, tgt_mask, tgt_target = src.to(device), src_mask.to(device), tgt_input.to(
            device), tgt_mask.to(device), tgt_target.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            pred = transformer(src, src_mask, tgt_input, tgt_mask)
            pred = pred.flatten(0, 1)
            tgt_target = tgt_target.flatten()

            loss = loss_fn(pred, tgt_target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses += loss.item()

    return losses / len(train_dataloader)


def train(args):
    dataset = read_dataset(args.dataset_path)
    d_model = args.d_model
    heads = args.heads
    max_len = args.max_len
    num_layers = args.num_layers
    batch_size = args.batch_size
    epochs = args.epochs
    resume = args.resume
    checkpoint_path = args.checkpoint_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (args.device == 'cpu'):
        device = torch.device("cpu")

    train_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True)

    dataset = ConversationDataset(dataset, max_len=max_len)
    transformer = Transformer(d_model, heads, num_layers, len(dataset.vocab_transform))

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
    scaler = GradScaler()
    if (resume):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        transformer.load_state_dict(checkpoint['model_state_dict'])
    transformer = transformer.to(device)

    for epoch in range(1, epochs + 1):
        print(f"Start epoch: {epoch}")

        start_time = timer()
        train_loss = train_epoch(train_dataloader, transformer, optimizer, loss_fn, device, scaler)
        end_time = timer()

        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

        state = {'model_state_dict': transformer.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
        torch.save(state, 'transformer.tar')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='qapairs.csv',
                        help='path to dataset, column names must be question and answer')
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--checkpoint_path", type=str, default='transformer.tar', help='path to read')
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--eps", type=float, default=1e-9)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    train(args)
