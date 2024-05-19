import pandas as pd

import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from config import PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX, special_symbols


class ConversationDataset(Dataset):
    def __init__(self, df_file, max_len=100, init_dataset=True) -> None:
        super().__init__()

        if init_dataset:
            self._init_dataset(df_file, max_len)

    def __len__(self):
        return len(self.src_batch)

    def __getitem__(self, idx):
        src, tgt = self.src_batch[idx], self.tgt_batch[idx]

        tgt_input = tgt[:-1]
        tgt_target = tgt[1:]

        src_mask, tgt_mask = self.create_src_mask(src), self.create_tgt_mask(tgt_input)
        return src, src_mask, tgt_input, tgt_mask, tgt_target

    def _init_dataset(self, df, max_len):
        print('read csv')
        df = df.astype(str)

        print('convert to list')
        question = df['question'].to_list()
        answer = df['answer'].to_list()
        words = question + answer

        print('build vocab & transform')
        self.token_transform = get_tokenizer('spacy', language='en_core_web_sm')
        self.vocab_transform = build_vocab_from_iterator(self.yield_tokens(words), min_freq=1, specials=special_symbols,
                                                         special_first=True)
        self.text_transform = self.sequential_transforms(self.token_transform, self.vocab_transform,
                                                         self.tensor_transform)

        print('set default index')
        self.vocab_transform.set_default_index(UNK_IDX)

        print('transform batch')
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in zip(question, answer):
            src_batch.append(self.text_transform(src_sample.rstrip("\n")))
            tgt_batch.append(self.text_transform(tgt_sample.rstrip("\n")))

        print('pad sequence')
        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)

        print('clip the length to fit data')
        self.src_batch = src_batch[:, :max_len]
        self.tgt_batch = tgt_batch[:, :max_len]

        print('finish init dataset')

    def yield_tokens(self, data_iter):
        for data_sample in data_iter:
            yield self.token_transform(data_sample)

    def sequential_transforms(self, *transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input

        return func

    def tensor_transform(self, token_ids):
        return torch.cat((torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX])))

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones((sz, sz))).transpose(0, 1)
        return mask

    def create_src_mask(self, src):
        src_seq_len = src.shape[-1]
        src_lookahead_mask = torch.ones((src_seq_len, src_seq_len)).bool()
        src_padding_mask = (src != PAD_IDX)
        src_mask = src_padding_mask.unsqueeze(0) & src_lookahead_mask

        return src_mask.unsqueeze(0)

    def create_tgt_mask(self, tgt):
        tgt_seq_len = tgt.shape[-1]
        tgt_lookahead_mask = self.generate_square_subsequent_mask(tgt_seq_len).bool()
        tgt_padding_mask = (tgt != PAD_IDX)
        tgt_mask = tgt_padding_mask.unsqueeze(0) & tgt_lookahead_mask

        return tgt_mask.unsqueeze(0)


def read_dataset(data_path):
    df = pd.read_csv(data_path)
    return df
