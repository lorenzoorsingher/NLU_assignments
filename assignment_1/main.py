import math
import os
import copy

import torch
import wandb
import json
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv


from time import time, localtime, strftime
from torch import nn, optim
from functools import partial

from models import LM_LSTM_TWO
from utils import Lang, read_file, get_vocab
from dataLoader import PennTreeBank, DataLoader, collate_fn
from setup import get_args
from utils import generate_id


def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:

        optimizer.zero_grad()  # Zeroing the gradient
        output = model(sample["source"])
        loss = criterion(output, sample["target"])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward()  # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()  # Update the weights

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return


def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad():  # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample["source"])
            loss = eval_criterion(output, sample["target"])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.xavier_uniform_(
                            param[idx * mul : (idx + 1) * mul]
                        )
                elif "weight_hh" in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.orthogonal_(param[idx * mul : (idx + 1) * mul])
                elif "bias" in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)


args = get_args()

FROM_JSON = args["json"]
TRAIN_BS = args["train_batch_size"]
DEV_BS = args["dev_batch_size"]
TEST_BS = args["test_batch_size"]

load_dotenv()

WANDB_SECRET = os.getenv("WANDB_SECRET")
wandb.login(key=WANDB_SECRET)


DEVICE = "cuda:0"  # it can be changed with 'cpu' if you do not have a gpu


train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

# Vocab is computed only on training set
# We add two special tokens end of sentence and padding
vocab = get_vocab(train_raw, ["<pad>", "<eos>"])

lang = Lang(train_raw, ["<pad>", "<eos>"])

train_dataset = PennTreeBank(train_raw, lang)
dev_dataset = PennTreeBank(dev_raw, lang)
test_dataset = PennTreeBank(test_raw, lang)

# Dataloader instantiation
print(f"Using Train BS: {TRAIN_BS}, Dev BS: {DEV_BS}, Test BS: {TEST_BS}")

train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BS,
    collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=DEVICE),
    shuffle=True,
)
dev_loader = DataLoader(
    dev_dataset,
    batch_size=DEV_BS,
    collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=DEVICE),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=TEST_BS,
    collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=DEVICE),
)


save_path = "assignment_1/checkpoints/"

if FROM_JSON:
    print("loading from json...")
    defaults, experiments = json.load(open("assignment_1/experiments.json"))
else:
    defaults = {
        "emb_size": 300,
        "hid_size": 300,
        "lr": 1.5,
        "clip": 5,
        "n_layers": 1,
        "emb_drop": 0.1,
        "out_drop": 0.1,
        "tying": False,
        "var_drop": False,
        "EPOCHS": 99,
        "OPT": "SGD",
    }
    experiments = [
        {
            "lr": 1.8,
            "emb_drop": 0.5,
            "out_drop": 0.5,
            "var_drop": True,
        },
        {
            "lr": 2.1,
            "emb_drop": 0.5,
            "out_drop": 0.5,
            "var_drop": True,
        },
        {
            "emb_drop": 0.1,
            "out_drop": 0.1,
            "var_drop": True,
        },
        {
            "emb_drop": 0.25,
            "out_drop": 0.0,
            "var_drop": True,
        },
    ]


for exp in experiments:

    args = defaults | exp

    print(args)

    emb_size = args["emb_size"]
    hid_size = args["hid_size"]
    lr = args["lr"]
    clip = args["clip"]
    n_layers = args["n_layers"]
    emb_drop = args["emb_drop"]
    out_drop = args["out_drop"]
    tying = args["tying"]
    var_drop = args["var_drop"]
    OPT = args["OPT"]
    device = "cuda:0"

    vocab_len = len(lang.word2id)

    model = LM_LSTM_TWO(
        emb_size,
        hid_size,
        vocab_len,
        tie=tying,
        out_dropout=out_drop,
        emb_dropout=emb_drop,
        n_layers=n_layers,
        var_drop=var_drop,
        pad_index=lang.word2id["<pad>"],
    ).to(device)

    model.apply(init_weights)

    if OPT == "SGD":
        print("using SGD")
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        print("using AdamW")
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(
        ignore_index=lang.word2id["<pad>"], reduction="sum"
    )

    # build run folder

    run_name = f"{emb_size}_{hid_size}_{int(emb_drop*100)}_{int(out_drop*100)}_{str(lr).replace('.','-')}"

    if var_drop:
        run_name += "_VD"
    if tying:
        run_name += "_TIE"
    if OPT == "AdamW":
        run_name += "_AdamW"
    else:
        run_name += "_SGD"

    run_path = f"{save_path + run_name}_{generate_id(5)}/"

    if os.path.exists(run_path):
        while os.path.exists(run_path):
            run_path = f"{save_path + run_name}_{generate_id(5)}/"

    os.mkdir(run_path)

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="NLU_assignment",
        name=run_name,
        config={
            "model": str(type(model).__name__),
            "lr": lr,
            "optim": str(type(optimizer).__name__),
            "clip": clip,
            "hid_size": hid_size,
            "emb_size": emb_size,
            "layers": n_layers,
            "tie": tying,
            "var_drop": var_drop,
            "dropout": [emb_drop, out_drop],
        },
    )

    EPOCHS = args["EPOCHS"]
    PAT = 3
    SAVE_RATE = 1
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    best_epoch = -1
    pbar = tqdm(range(1, EPOCHS))

    for epoch in pbar:
        ppl_train, loss = train_loop(
            train_loader, optimizer, criterion_train, model, clip
        )

        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())

            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())

            if ppl_dev < best_ppl:  # the lower, the better
                best_ppl = ppl_dev
                best_epoch = epoch
                patience = PAT
                best_model = copy.deepcopy(model).to("cpu")
            else:
                patience -= 1

            pbar.set_description(
                "PPL: "
                + str(round(ppl_dev, 2))
                + " best: "
                + str(round(best_ppl, 2))
                + " P: "
                + str(patience)
            )
            wandb.log({"ppl": ppl_dev, "ppl_train": ppl_train, "loss": loss_dev})

        if epoch % SAVE_RATE == 0:
            checkpoint_path = run_path + "epoch_" + ("0000" + str(epoch))[-4:] + ".pt"
            torch.save(model.state_dict(), checkpoint_path)

        if patience <= 0:  # Early stopping with patience
            break  # Not nice but it keeps the code clean

    checkpoint_path = run_path + "epoch_" + ("0000" + str(epoch))[-4:] + ".pt"
    torch.save(model.state_dict(), checkpoint_path)

    best_model.to(device)
    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)

    print("Best ppl: ", best_ppl)
    print("Test ppl: ", final_ppl)