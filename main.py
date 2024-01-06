import torch
import time

from torch import Tensor
from rich.progress import Progress
from torch.utils.data import DataLoader, random_split
from dataset import PoetryData
from model import PoetryNet

batch_size = 48
lr = 0.001


class PoetryGen:
    def __init__(self) -> None:
        # self.device = (
        #     torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        # )
        self.device = torch.device("cuda:0")
        self.dataset = PoetryData(self.device, max_lines=3000)
        self.vocab_size = self.dataset.vocab_size
        train_data, test_data = random_split(
            self.dataset, [len(self.dataset) - 1000, 1000]
        )
        self.train_dataloader = DataLoader(train_data, batch_size, True)
        self.test_dataloader = DataLoader(test_data, batch_size, True)

        self.net = PoetryNet(self.vocab_size, self.device, embed_size=128).to(
            self.device
        )
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr)
        self.loss_f = torch.nn.CrossEntropyLoss(ignore_index=2)
        self.loaded_checkpoint_file = None
        self.epoch = 0

        self.progress = Progress()

        import glob

        files = glob.glob("checkpoint-*.pth")
        for i, file in enumerate(files):
            print(f"{i}> {file}")
        if files:
            t = input(
                "choose check point to load, default is the last one, n to unload>"
            )
            if t == "":
                t = -1
            if t != "n":
                self.load_checkpoint(files[int(t)])

    def save_checkpoint(self):
        file_name = (
            self.loaded_checkpoint_file
            or f'checkpoint-{time.strftime("%y%m%d-%H%M")}.pth'
        )
        with open(file_name, "wb") as file:
            torch.save(
                {
                    "net_state": self.net.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "epoch": self.epoch,
                },
                file,
            )
        print(f"save check point to {file_name}")
        self.loaded_checkpoint_file = file_name

    def load_checkpoint(self, file: str):
        ckpt = torch.load(file)
        self.net.load_state_dict(ckpt["net_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.epoch = ckpt["epoch"]

        self.loaded_checkpoint_file = file
        print(f"loaded check point: {file}")

    def generate(self):
        self.net.eval()
        src = self.dataset.word2token("风细孤帆雨后低").unsqueeze(0)
        tgt = torch.LongTensor([[0]]).to(self.device)
        memo = self.net.encode(src)
        res = []
        for i in range(48):
            out = self.net.decode(tgt, memo)
            next_word = out.argmax(2)
            if next_word[0][-1] == 1:
                break
            # tgt = torch.cat((tgt, ))
            res.append(next_word[0][-1].item())
            tgt = torch.cat((tgt, next_word[:, -1:]), 1)

        return self.dataset.token2word(res)

    def forward_net(self, src: Tensor, tgt: Tensor):
        src, tgt = src.to(self.device), tgt.to(self.device)
        src_mask = (src == 2).to(self.device)

        dec_tgt = tgt[:, :-1]
        dec_tgt_mask = (dec_tgt == 2).to(self.device)
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            dec_tgt.size(1), self.device
        )

        out = self.net.forward(src, dec_tgt, tgt_mask, src_mask, dec_tgt_mask)
        return out

    def train_epoch(self):
        self.net.train()

        # ignore <pad> which index is 2
        loss_f = self.loss_f  # torch.nn.CrossEntropyLoss(ignore_index=2)

        voacb_size = self.dataset.vocab_size
        for i, (src, tgt) in enumerate(self.train_dataloader):
            out = self.forward_net(src, tgt)
            # out = torch.max(out, 2).indices
            loss = loss_f.forward(out.reshape(-1, voacb_size), tgt[:, 1:].flatten())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % 40 == 0:
                print(f"epoch={self.epoch} loss={loss.item():.4f} {i}")
                # print(self.dataset.token2word(tgt[0]))
                # print(self.dataset.token2word(out[0].argmax(1).tolist()))

    def evaluation(self):
        self.net.eval()

        loss_f = self.loss_f
        voacb_size = self.dataset.vocab_size

        loss_a = 0
        with torch.no_grad():
            for i, (src, tgt) in enumerate(self.test_dataloader):
                out = self.forward_net(src, tgt)
                # out = torch.max(out, 2).indices
                loss = loss_f.forward(out.reshape(-1, voacb_size), tgt[:, 1:].flatten())
                loss_a += loss.item()

        print(
            f"Validation: epoch={self.epoch} avg loss={loss_a/len(self.test_dataloader):.4f}"
        )

    def training(self, train_epoch_nums: int = 36):
        self.progress.start()
        for i in range(train_epoch_nums):
            self.train_epoch()
            self.evaluation()
            self.epoch += 1
            self.save_checkpoint()
            print(self.generate())


def main():
    model = PoetryGen()
    # print(model.generate())
    model.training(1024)


if __name__ == "__main__":
    main()
