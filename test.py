import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from config import *
from sklearn.metrics import classification_report


from model import build_transformer
from dataset import TrafficDataset
from utils import seed_everything, compute_emb




def run(test_dataloader, loss_fn, model, header_model, pack_model, header_pack_model, config):
    
    with torch.no_grad():
        model.eval()
        header_model.eval()
        pack_model.eval()
        header_pack_model.eval()

        test_loss, preds, labels = [], [], []
        for batch in test_dataloader:
            seq, label, mask, seq_header, seq_header_mask = batch
            batch_size = seq.shape[0]

            seq, label, mask, seq_header, seq_header_mask = seq.to(config.device), label.to(config.device), mask.to(config.device), seq_header.to(config.device), seq_header_mask.to(config.device)
            
            seq_emb = compute_emb(model, pack_model, seq, mask, config.BYTE_PAD_TRUNC_LENGTH, seq_byte_mask = None, baseline = config.baseline)

            header_emb = compute_emb(header_model, header_pack_model, seq_header, seq_header_mask, config.HEADER_BYTE_PAD_TRUNC_LENGTH, seq_byte_mask = None, baseline = config.baseline)


            emb = torch.cat([seq_emb, header_emb], dim = 1) # (batch, 2*d_model)
            
            
            pred = model.project(emb) # (batch, class_size)
            loss = loss_fn(pred, label)

            test_loss.append(loss.item())
            preds.append(pred)
            labels.append(label)
    
    test_loss = np.mean(test_loss)
    #calculate acc
    preds = torch.cat(preds, dim = 0)
    labels = torch.cat(labels, dim = 0)
    
    print(classification_report(labels.cpu().numpy(), preds.argmax(dim = 1).cpu().numpy(), digits=4))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['iscx-vpn', 'iscx-nonvpn', 'iscx-tor', 'iscx-nontor', 'ios', 'android', 'ios-cross-plat', 'android-cross-plat'], required=True)
    parser.add_argument("--baseline", type=str, default='Byte-Transformer', choices=['Byte-Transformer', 'Byte-Transformer wo Byte-att', 'Byte-Transformer wo Byte-Pack-att', 'ET-Bert'])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    opt = parser.parse_args()

    config = get_config(opt.dataset)
    config.baseline = opt.baseline
    if opt.device is not None:
        config.device = opt.device
    elif not torch.cuda.is_available():
        config.device = 'cpu'
    if opt.batch_size is not None:
        config.batch_size = opt.batch_size
    seed_everything(config.SEED)


    header_model, header_pack_model = build_transformer(config.seq_vocab_size, config.seq_seq_len, config.seq_pack_len, config.class_size, header = True)
    model, pack_model = build_transformer(config.seq_vocab_size, config.seq_seq_len, config.seq_pack_len, config.class_size)

    model.to(config.device)
    header_model.to(config.device)
    pack_model.to(config.device)
    header_pack_model.to(config.device)

    def has_checkpoints(path):
        return all(
            os.path.exists(os.path.join(path, filename))
            for filename in ('model.pth', 'model_pack.pth', 'model_header.pth', 'model_header_pack.pth')
        )

    checkpoint_dir = f'./model/{config.baseline}/{config.dataset}'
    if not has_checkpoints(checkpoint_dir):
        fallback_roots = [
            os.environ.get('TRAFFIC_EXPLAINER_CHECKPOINT_ROOT'),
            '../Traffic-Explainer-checkpoints',
        ]
        for fallback_root in fallback_roots:
            if not fallback_root:
                continue
            fallback_dir = os.path.join(fallback_root, config.baseline, config.dataset)
            if has_checkpoints(fallback_dir):
                checkpoint_dir = fallback_dir
                break

    map_location = torch.device(config.device)

    def load_state(path):
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=map_location)

    model.load_state_dict(load_state(f'{checkpoint_dir}/model.pth'))
    header_model.load_state_dict(load_state(f'{checkpoint_dir}/model_header.pth'))
    pack_model.load_state_dict(load_state(f'{checkpoint_dir}/model_pack.pth'))
    header_pack_model.load_state_dict(load_state(f'{checkpoint_dir}/model_header_pack.pth'))
    
    test_dataset = TrafficDataset(config.dataset, flag = 'test', PAD_TRUNC_DIGIT = config.PAD_TRUNC_DIGIT)
    test_dataloader = DataLoader(test_dataset, batch_size = config.batch_size, shuffle=False)
    loss_fn = nn.CrossEntropyLoss().to(config.device)

    run(test_dataloader, loss_fn, model, header_model, pack_model, header_pack_model, config)
