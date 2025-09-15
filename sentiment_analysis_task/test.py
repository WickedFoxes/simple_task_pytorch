import os
import time
import argparse
import torch

from utils.datasets import get_imdb_test_dataloader
from utils.evaluate import accuracy
from utils.loss import build_loss
from utils.util import set_seed

from models import LSTMClassifier

from transformers import AutoTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='lstm_classification', type=str, help='model name')
    parser.add_argument('--chk_path', type=str, required=True, help='the checkpoint file you want to test')
    parser.add_argument('--data_dir', default='./data', type=str,
                    help='path to dataset')
    parser.add_argument('--dataset', default='imdb', type=str,
                        help='dataset name')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='batch size')
    parser.add_argument('--seed', default=42, type=int,
                        help='random seed')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of workers for data loading')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--embed_dim', default=300, type=int,
                        help='embedding dimension')
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help='hidden dimension')
    parser.add_argument('--num_layers', default=2, type=int,
                        help='number of layers')
    parser.add_argument('--bidirectional', action='store_true',
                        help='use bidirectional LSTM')
    parser.add_argument('--dropout', default=0.3, type=float,
                        help='dropout rate')
    parser.add_argument('--max_len', default=400, type=int,
                        help='maximum sequence length')
    parser.add_argument('--loss', default='CE', type=str,
                        help='loss function')
    args = parser.parse_args()

    ckpt_path = args.chk_path
    loss_type = args.loss
    model_name = args.model
    data_dir = args.data_dir
    dataset = args.dataset
    batch_size = args.batch_size
    seed = args.seed
    num_workers = args.num_workers
    num_classes = args.num_classes
    embed_dim = args.embed_dim
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    bidirectional = args.bidirectional
    dropout = args.dropout
    max_len = args.max_len

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab = tokenizer.get_vocab()
    pad_idx = vocab["[PAD]"]


    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if dataset == 'imdb':
        test_loader = get_imdb_test_dataloader(
            tokenizer,
            data_path=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            max_len=max_len
        )

    if model_name == 'lstm_classification':
        model = LSTMClassifier(
            vocab_size=len(vocab),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            pad_idx=pad_idx
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 손실 함수 빌드
    criterion = build_loss(loss_type)

    # 모델을 디바이스로 이동
    model = model.to(device)
    model.eval()
    running_loss, running_acc, n = 0.0, 0.0, 0
    
    with torch.no_grad():
        for input_ids, lengths, labels in test_loader:
            input_ids = input_ids.to(device)
            lengths   = lengths.to(device)
            targets    = labels.to(device)

            logits = model(input_ids, lengths)
            loss = criterion(logits, targets)

            bs = input_ids.size(0)
            running_loss += loss.item() * bs
            running_acc  += accuracy(logits, targets) * bs
            n += bs
    print(f"Test Acc: {running_acc / n}")
    print(f"Test Loss: {running_loss / n}")