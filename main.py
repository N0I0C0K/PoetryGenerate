import torch
import time

from torch import Tensor
from rich.progress import Progress
from torch.utils.data import DataLoader, random_split
from dataset import PoetryData
from model import PoetryNet

batch_size = 64
lr = 0.0001


class PoetryGen:
    def __init__(self) -> None:
        # self.device = (
        #     torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        # )
        self.device = torch.device("cuda:0")
        self.dataset = PoetryData(self.device, max_lines=50000, token_length=12)
        self.vocab_size = self.dataset.vocab_size
        train_data, test_data = random_split(
            self.dataset, [len(self.dataset) - 1000, 1000]
        )
        self.train_dataloader = DataLoader(train_data, batch_size, True)
        self.test_dataloader = DataLoader(test_data, batch_size, True)

        self.net = PoetryNet(self.vocab_size, self.device, embed_size=512).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr)
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 256
        )

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
        self.optimizer_scheduler.last_epoch = self.epoch
        print(f"loaded check point: {file}, epoch: {self.epoch}")

    def generate_one(self, pre_sentence: str, start_words: str = ""):
        self.net.eval()
        start_words_token = [0]
        start_words_token.extend(self.dataset.word2idx[x] for x in start_words)
        src = self.dataset.word2token(pre_sentence).unsqueeze(0)
        tgt = torch.LongTensor([start_words_token]).to(self.device)
        memo = self.net.encode(src)
        res = []
        for i in range(12):
            out = self.net.decode(tgt, memo)
            next_word = out.argmax(2)
            if next_word[0][-1] == 1:
                break
            # tgt = torch.cat((tgt, ))
            res.append(next_word[0][-1].item())
            tgt = torch.cat((tgt, next_word[:, -1:]), 1)

        return start_words + self.dataset.token2word(res)

    def generate(self, num_sentence: int, pre_style: str):
        res = []
        for i in range(num_sentence):
            s = self.generate_one(pre_style if not res else res[-1])
            res.append(s)
        return "/".join(res)

    def generate_by_start(self, start_words: str, pre_style: str) -> str:
        """generate sentence by start words

        Args:
            start_words (str): start words for ever sentence, Divide by /, for example: 我/你/他/你
            pre_style(str): style for the poem
        Returns:
            str: the result
        """
        res = []
        start_words_l = start_words.split("/")
        if not start_words_l:
            return ""
        for i, s in enumerate(start_words_l):
            t = self.generate_one(pre_style if not res else res[-1], s)
            res.append(t)
        return "/".join(res)

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
        train_progress = self.progress.add_task(
            description="Train Epoch", total=len(self.train_dataloader)
        )
        # ignore <pad> which index is 2
        loss_f = self.loss_f  # torch.nn.CrossEntropyLoss(ignore_index=2)

        voacb_size = self.dataset.vocab_size
        len_data = len(self.train_dataloader)
        loss_all = 0
        for i, (src, tgt) in enumerate(self.train_dataloader):
            out = self.forward_net(src, tgt)
            # out = torch.max(out, 2).indices
            loss = loss_f.forward(out.reshape(-1, voacb_size), tgt[:, 1:].flatten())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.progress.update(
                train_progress,
                advance=1,
                description=f"{i}/{len_data} loss={loss.item():.4f}",
            )
            loss_all += loss.item()
            # if i % 40 == 0:
            #     print(f"epoch={self.epoch} loss={loss.item():.4f} {i}")
            #     # print(self.dataset.token2word(tgt[0]))
            #     # print(self.dataset.token2word(out[0].argmax(1).tolist()))
        self.optimizer_scheduler.step()
        self.progress.remove_task(train_progress)
        self.progress.print(
            f"train epoch={self.epoch} average loss={loss_all/len_data:.4f} lr={self.optimizer_scheduler.get_lr()}"
        )

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

        self.progress.print(
            f"Validation: epoch={self.epoch} avg loss={loss_a/len(self.test_dataloader):.4f}"
        )

    def training(self, train_epoch_nums: int = 36):
        self.progress.start()
        training_all = self.progress.add_task(
            description=f"epoch={self.epoch} lr={self.optimizer_scheduler.get_lr()}",
            total=train_epoch_nums,
        )
        for i in range(train_epoch_nums):
            self.progress.update(
                training_all,
                advance=1,
                description=f"epoch={self.epoch} lr={self.optimizer_scheduler.get_lr()}",
            )
            self.train_epoch()
            self.evaluation()
            self.epoch += 1
            self.save_checkpoint()
            print(self.generate(4, "绿蔓如藤不用栽"))


def main():
    model = PoetryGen()
    # print(model.generate())
    # while (s := input(">")) != "exit":
    #     print(model.generate(4, "床前明月光"))
    model.training(256)


if __name__ == "__main__":
    main()
