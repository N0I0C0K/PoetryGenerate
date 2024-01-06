import re
import torch
from torch.utils.data import Dataset
from torch import device
from torch.nn.functional import pad


class PoetryData(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        device: device,
        *,
        token_length: int = 48,
        poetry_file: str = "./archive/chinese_poems.txt",
        max_lines: int = 12000,
    ) -> None:
        super().__init__()
        self.corpus: list[tuple[str, str]] = []
        self.token_length = token_length
        self.idx2word = ["<bos>", "<eos>", "<pad>"]
        self.word2idx = {v: k for k, v in enumerate(self.idx2word)}
        idx = 3
        loaded_lines = 0
        # 绿蔓如藤不用栽,淡青花遶竹篱开.披衣向晓还堪爱,忽见蜻蜓带露来.
        self.device = device
        with open(poetry_file, "r", encoding="utf-8") as file:
            while loaded_lines < max_lines or max_lines == -1:
                line = file.readline().strip(" \n\r")
                if len(line) == 0:
                    break
                loaded_lines += 1
                # self.corpus.append(line)
                for k in line:
                    if k not in self.word2idx:
                        self.word2idx[k] = idx
                        self.idx2word.append(k)
                        idx += 1
                for pair in line.split("."):
                    t = pair.split(",")
                    if len(t) == 2:
                        self.corpus.append((t[0], t[1]))

        self.vocab_size = len(self.word2idx)

    def word2token(self, words: str) -> torch.Tensor:
        t = [0]
        t.extend(self.word2idx[x] for x in words[: self.token_length - 2])
        t.append(1)
        t.extend(2 for _ in range(max(0, self.token_length - len(t))))
        return torch.LongTensor(t).to(self.device)

    def token2word(self, tokens: list[int]) -> str:
        return "".join(self.idx2word[x] for x in tokens)

    def get_token_mask(self, token: torch.Tensor) -> torch.Tensor:
        return (token == 2).to(self.device)

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        up, down = self.corpus[index]
        token = self.word2token(up)
        token_res = self.word2token(down)
        return (token, token_res)
