import argparse

from data_utils import read_dataset, ConversationDataset
from config import BOS_IDX, EOS_IDX
import torch

from model import Transformer


def predict_answer(question, transformer, dataset, device, max_len):
    src = dataset.text_transform(question).unsqueeze(0).to(device)  # Move to device here
    src_mask = dataset.create_src_mask(src).to(device)  # Move to device here
    src = src

    encoded = transformer.encode(src, src_mask)
    start_y = torch.tensor([[BOS_IDX]]).type_as(src).to(device)  # Move to device here
    ys = start_y

    for i in range(max_len - 1):
        tgt_mask = dataset.create_tgt_mask(ys).to(device)  # Move to device here
        logit = transformer.decode(ys, tgt_mask, encoded)
        next_word = logit.argmax(-1)

        ys = torch.cat([start_y, next_word], dim=1)
        if next_word[0, -1].item() == EOS_IDX:
            break

    ys = ys.flatten().cpu().tolist()  # Move to CPU for processing
    ys = " ".join(dataset.vocab_transform.lookup_tokens(ys)).replace("<bos>", "").replace("<eos>", "")
    return ys


def inference(args):
    df = read_dataset(args.dataset_path)
    dataset = ConversationDataset(df)
    checkpoint_path = args.checkpoint_path

    checkpoint = torch.load(checkpoint_path)
    max_len = args.max_len
    d_model = args.d_model
    heads = args.heads
    num_layers = args.num_layers

    transformer = Transformer(d_model, heads, num_layers, len(dataset.vocab_transform))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == 'cpu':
        device = torch.device("cpu")

    transformer.load_state_dict(checkpoint['model_state_dict'])
    transformer.to(device)
    transformer.eval()
    while 1:
        print('----------')
        question = input("Question: ")
        if question == 'quit':
            exit()
        sentence = predict_answer(question, transformer, dataset, device, max_len)
        print("Answer: " + sentence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="qapairs.csv")
    parser.add_argument("--checkpoint_path", type=str, default='transformer.tar', help='path to read')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    args = parser.parse_args()
    inference(args)
