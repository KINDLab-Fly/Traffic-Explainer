import argparse
import os
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import get_config
from dataset import TrafficDatasetClass
from explain import compute_metrics, evaluate_selected_bytes, load_models, parse_budgets, print_metrics, predict_logits, topk_from_scores
from model import explainer_model
from utils import budget_constrain, cal_class_entropy_class, seed_everything


EXPLANATION_ALIASES = {
    'traffic-explainer': 'byte-level',
    'byte-level': 'byte-level',
    'random': 'random',
    'saliency': 'saliency-map',
    'saliency-map': 'saliency-map',
    'lime': 'lime',
}


def class_labels(config):
    labels = np.load(f'./dataset/{config.dataset}/train_pyg.npz')['label']
    return np.unique(labels)


def predict_by_class(test_dataloaders, model, header_model, pack_model, header_pack_model, config):
    pred_labels_by_class = defaultdict(list)
    ground_truths = []
    pred_labels = []

    for class_pos, test_dataloader in enumerate(test_dataloaders):
        for batch in tqdm(test_dataloader, desc=f'predict-class-{class_pos}'):
            seq, label, mask, seq_header, seq_header_mask = batch
            seq = seq.to(config.device, non_blocking=True)
            label = label.to(config.device, non_blocking=True)
            mask = mask.to(config.device, non_blocking=True)
            seq_header = seq_header.to(config.device, non_blocking=True)
            seq_header_mask = seq_header_mask.to(config.device, non_blocking=True)

            with torch.no_grad():
                pred = predict_logits(model, header_model, pack_model, header_pack_model, seq, mask, seq_header, seq_header_mask, config)
            pred_label = pred.argmax(dim=1).detach().cpu().numpy()
            pred_labels_by_class[class_pos].extend(pred_label)
            ground_truths.extend(label.detach().cpu().numpy())
            pred_labels.extend(pred_label)

    return pred_labels_by_class, ground_truths, pred_labels


def class_saliency_scores(test_dataloader, model, header_model, pack_model, header_pack_model, config):
    model.zero_grad(set_to_none=True)
    header_model.zero_grad(set_to_none=True)
    pack_model.zero_grad(set_to_none=True)
    header_pack_model.zero_grad(set_to_none=True)

    for batch in test_dataloader:
        seq, label, mask, seq_header, seq_header_mask, pred_label = batch
        seq = seq.to(config.device, non_blocking=True)
        mask = mask.to(config.device, non_blocking=True)
        seq_header = seq_header.to(config.device, non_blocking=True)
        seq_header_mask = seq_header_mask.to(config.device, non_blocking=True)

        pred = predict_logits(model, header_model, pack_model, header_pack_model, seq, mask, seq_header, seq_header_mask, config)
        probs = softmax(pred, dim=1)
        pred_label = pred.argmax(dim=1).detach()
        probs[torch.arange(probs.size(0), device=config.device), pred_label].mean().backward()

    grad_payload = torch.norm(model.seq_embed.embedding.weight.grad, dim=1)
    grad_header = torch.norm(header_model.seq_embed.embedding.weight.grad, dim=1)
    scores = 0.5 * (grad_payload + grad_header)
    return scores.detach(), scores.detach()


def class_lime_scores(test_dataloader, model, header_model, pack_model, header_pack_model, class_label, config):
    payload_scores_np = np.zeros(config.seq_vocab_size, dtype=np.float32)
    header_scores_np = np.zeros(config.seq_vocab_size, dtype=np.float32)
    weight_sum = 0.0
    taken = 0
    payload_len = config.BYTE_PAD_TRUNC_LENGTH
    header_len = config.HEADER_BYTE_PAD_TRUNC_LENGTH
    total_feats = payload_len + header_len
    baseline_token = config.PAD_TRUNC_DIGIT
    baseline_vector = np.full((1, total_feats), fill_value=baseline_token, dtype=np.int64)
    kernel_width = 0.75 * np.sqrt(total_feats)

    for batch in test_dataloader:
        seq, label, mask, seq_header, seq_header_mask, pred_label = batch
        batch_size = seq.shape[0]
        for idx in range(batch_size):
            if taken >= config.class_lime_max_samples:
                break

            sample_seq = seq[idx:idx + 1].to(config.device)
            sample_mask = mask[idx:idx + 1].to(config.device)
            sample_header = seq_header[idx:idx + 1].to(config.device)
            sample_header_mask = seq_header_mask[idx:idx + 1].to(config.device)

            with torch.no_grad():
                pred = predict_logits(model, header_model, pack_model, header_pack_model, sample_seq, sample_mask, sample_header, sample_header_mask, config)
                probs = softmax(pred, dim=1)
            pred_class = int(probs.argmax(dim=1).item())
            if pred_class != int(class_label):
                continue

            sample_vector = np.concatenate(
                [
                    sample_seq.detach().cpu().numpy().reshape(-1)[:payload_len],
                    sample_header.detach().cpu().numpy().reshape(-1)[:header_len],
                ]
            ).reshape(1, -1)
            seq_tokens = sample_vector[0, :payload_len]
            header_tokens = sample_vector[0, payload_len:payload_len + header_len]

            orig_mask = (sample_vector != baseline_token).astype(np.int64).reshape(-1)
            z_mask = np.random.randint(0, 2, size=(config.class_lime_perturbations, total_feats), dtype=np.int64)
            z_mask = z_mask * orig_mask
            z_mask[0] = orig_mask

            x_perturbed = np.tile(baseline_vector, (config.class_lime_perturbations, 1))
            rows, cols = np.where(z_mask == 1)
            x_perturbed[rows, cols] = sample_vector[0, cols]

            x_seq = torch.as_tensor(x_perturbed[:, :payload_len], dtype=torch.long, device=config.device)
            x_header = torch.as_tensor(x_perturbed[:, payload_len:payload_len + header_len], dtype=torch.long, device=config.device)
            x_seq_mask = (x_seq == baseline_token).long()
            x_header_mask = (x_header == baseline_token).long()

            with torch.no_grad():
                pred_probs = softmax(
                    predict_logits(
                        model,
                        header_model,
                        pack_model,
                        header_pack_model,
                        x_seq,
                        x_seq_mask,
                        x_header,
                        x_header_mask,
                        config,
                    ),
                    dim=1,
                ).detach().cpu().numpy()
            y = pred_probs[:, pred_class]

            diff = np.abs(z_mask - orig_mask)
            dist = diff.sum(axis=1) / (orig_mask.sum() + 1e-10)
            weights = np.exp(-(dist ** 2) / (kernel_width ** 2)).astype(np.float32)
            x_aug = np.hstack([np.ones((config.class_lime_perturbations, 1), dtype=np.float32), z_mask.astype(np.float32)])
            sqrt_w = np.sqrt(weights)
            coef_full, *_ = np.linalg.lstsq(x_aug * sqrt_w[:, np.newaxis], y * sqrt_w, rcond=None)
            coef = coef_full[1:]

            sample_payload_scores = np.zeros(config.seq_vocab_size, dtype=np.float32)
            sample_header_scores = np.zeros(config.seq_vocab_size, dtype=np.float32)

            for token in np.unique(seq_tokens):
                if token == baseline_token:
                    continue
                positions = np.where(seq_tokens == token)[0]
                if positions.size == 0:
                    continue
                weight = float(np.max(coef[positions]))
                if weight > 0 and weight > sample_payload_scores[int(token)]:
                    sample_payload_scores[int(token)] = weight

            for token in np.unique(header_tokens):
                if token == baseline_token:
                    continue
                positions = np.where(header_tokens == token)[0]
                if positions.size == 0:
                    continue
                weight = float(np.max(coef[payload_len + positions]))
                if weight > 0 and weight > sample_header_scores[int(token)]:
                    sample_header_scores[int(token)] = weight

            sample_weight = (
                float(probs[0, pred_class].item())
                if config.class_lime_use_conf_weight
                else 1.0
            )
            payload_scores_np += sample_weight * sample_payload_scores
            header_scores_np += sample_weight * sample_header_scores
            weight_sum += sample_weight
            taken += 1
        if taken >= config.class_lime_max_samples:
            break

    if weight_sum > 0:
        payload_scores_np /= weight_sum
        header_scores_np /= weight_sum
    return (
        torch.from_numpy(payload_scores_np).to(config.device),
        torch.from_numpy(header_scores_np).to(config.device),
    )


def traffic_explainer_class_scores(test_dataloader, model, header_model, pack_model, header_pack_model, config):
    explainer = explainer_model(config).to(config.device)
    explainer.train()
    optimizer = optim.Adam(explainer.parameters(), lr=config.explainer_lr)
    losses = []

    for epoch in tqdm(range(config.class_explainer_epochs), desc='class-explainer'):
        epoch_losses = []
        for batch in test_dataloader:
            seq, label, mask, seq_header, seq_header_mask, pred_label = batch
            seq = seq.to(config.device, non_blocking=True)
            mask = mask.to(config.device, non_blocking=True)
            seq_header = seq_header.to(config.device, non_blocking=True)
            seq_header_mask = seq_header_mask.to(config.device, non_blocking=True)
            pred_label = pred_label.to(config.device, non_blocking=True)

            pred = explainer(model, header_model, pack_model, header_pack_model, seq, mask, seq_header, seq_header_mask)
            loss_explain = cal_class_entropy_class(pred, pred_label)
            loss_budget = budget_constrain(explainer.mask_header, explainer.mask_payload, config.mask_budget, config.mask_budget)
            loss = loss_explain + loss_budget
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_losses.append(loss.item())

        if epoch_losses:
            losses.append(float(np.mean(epoch_losses)))
        if len(losses) > 5 and abs(losses[-1] - losses[-5]) < config.class_early_stop_delta:
            break

    return torch.sigmoid(explainer.mask_payload).detach(), torch.sigmoid(explainer.mask_header).detach()


def run(test_dataloaders, unique_labels, model, header_model, pack_model, header_pack_model, config):
    model.eval()
    header_model.eval()
    pack_model.eval()
    header_pack_model.eval()

    pred_labels_by_class, ground_truths, pred_labels = predict_by_class(
        [DataLoader(loader.dataset, batch_size=config.class_eval_batch_size, shuffle=False) for loader in test_dataloaders],
        model,
        header_model,
        pack_model,
        header_pack_model,
        config,
    )

    test_dataloaders = [
        DataLoader(dataset, batch_size=config.class_batch_size, shuffle=False)
        for idx, class_label in enumerate(unique_labels)
        for dataset in [
            TrafficDatasetClass(
                config.dataset,
                flag='test',
                class_idx=class_label,
                PAD_TRUNC_DIGIT=config.PAD_TRUNC_DIGIT,
                pred_label=pred_labels_by_class[idx],
            )
        ]
    ]
    if config.max_samples_per_class is not None:
        test_dataloaders = [
            DataLoader(
                Subset(loader.dataset, range(min(config.max_samples_per_class, len(loader.dataset)))),
                batch_size=config.class_batch_size,
                shuffle=False,
            )
            for loader in test_dataloaders
        ]

    pred_rm_labels = defaultdict(list)
    pred_add_labels = defaultdict(list)
    explain_header_idxs = [defaultdict(list) for _ in range(len(unique_labels))]
    explain_payload_idxs = [defaultdict(list) for _ in range(len(unique_labels))]

    for class_pos, (test_dataloader, class_label) in enumerate(zip(test_dataloaders, unique_labels)):
        print(f'===========Class_label:{class_label}===========')
        if config.explain_strategy == 'byte-level':
            payload_scores, header_scores = traffic_explainer_class_scores(test_dataloader, model, header_model, pack_model, header_pack_model, config)
        elif config.explain_strategy == 'random':
            payload_scores = header_scores = None
        elif config.explain_strategy == 'saliency-map':
            payload_scores, header_scores = class_saliency_scores(test_dataloader, model, header_model, pack_model, header_pack_model, config)
        elif config.explain_strategy == 'lime':
            payload_scores, header_scores = class_lime_scores(test_dataloader, model, header_model, pack_model, header_pack_model, class_label, config)
        else:
            raise ValueError(f'Unsupported class explanation: {config.explain_strategy}')

        for budget in config.eval_budgets:
            if config.explain_strategy == 'random':
                top_header = torch.randint(0, 257, (int(257 * budget),))
                top_payload = torch.randint(0, 257, (int(257 * budget),))
            else:
                top_header = topk_from_scores(header_scores, budget)
                top_payload = topk_from_scores(payload_scores, budget)

            for batch in test_dataloader:
                seq, label, mask, seq_header, seq_header_mask, pred_label = batch
                seq = seq.to(config.device, non_blocking=True)
                mask = mask.to(config.device, non_blocking=True)
                seq_header = seq_header.to(config.device, non_blocking=True)
                seq_header_mask = seq_header_mask.to(config.device, non_blocking=True)
                for sample_idx in range(seq.shape[0]):
                    pred_rm, pred_add = evaluate_selected_bytes(
                        model,
                        header_model,
                        pack_model,
                        header_pack_model,
                        seq[sample_idx:sample_idx + 1],
                        mask[sample_idx:sample_idx + 1],
                        seq_header[sample_idx:sample_idx + 1],
                        seq_header_mask[sample_idx:sample_idx + 1],
                        top_payload,
                        top_header,
                        config,
                    )
                    pred_rm_labels[budget].append(pred_rm)
                    pred_add_labels[budget].append(pred_add)

            explain_header_idxs[class_pos][budget] = top_header.detach().cpu().tolist()
            explain_payload_idxs[class_pos][budget] = top_payload.detach().cpu().tolist()

    out_dir = f'./res/{config.baseline}/{config.dataset}/class-level/{config.explain_strategy}'
    os.makedirs(out_dir, exist_ok=True)
    torch.save(ground_truths, f'{out_dir}/ground_truths.pt')
    torch.save(pred_labels, f'{out_dir}/pred_labels.pt')
    torch.save(dict(pred_rm_labels), f'{out_dir}/pred_rm_labels.pt')
    torch.save(dict(pred_add_labels), f'{out_dir}/pred_add_labels.pt')
    torch.save(explain_header_idxs, f'{out_dir}/explain_header_idxs.pt')
    torch.save(explain_payload_idxs, f'{out_dir}/explain_payload_idxs.pt')

    metrics = compute_metrics(ground_truths, pred_labels, pred_rm_labels, pred_add_labels)
    print_metrics(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['iscx-vpn', 'iscx-nonvpn', 'iscx-tor', 'iscx-nontor'], required=True)
    parser.add_argument('--baseline', type=str, default='Byte-Transformer')
    parser.add_argument('--explanation', type=str, choices=list(EXPLANATION_ALIASES.keys()), required=True)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--budgets', type=str, default='0.01,0.05,0.1')
    parser.add_argument('--mask_budget', type=float, default=0.1)
    parser.add_argument('--class_explainer_epochs', type=int, default=1000)
    parser.add_argument('--class_early_stop_delta', type=float, default=1e-4)
    parser.add_argument('--class_batch_size', type=int, default=32)
    parser.add_argument('--class_eval_batch_size', type=int, default=32)
    parser.add_argument('--explainer_lr', type=float, default=0.01)
    parser.add_argument('--lime_samples', type=int, default=30)
    parser.add_argument('--lime_alpha', type=float, default=0.1)
    parser.add_argument('--class_lime_max_samples', type=int, default=16)
    parser.add_argument('--class_lime_perturbations', type=int, default=200)
    parser.add_argument('--class_lime_use_conf_weight', type=int, default=1)
    parser.add_argument('--max_samples_per_class', type=int, default=None)
    opt = parser.parse_args()

    config = get_config(opt.dataset)
    config.baseline = opt.baseline
    config.explain_strategy = EXPLANATION_ALIASES[opt.explanation]
    config.eval_budgets = parse_budgets(opt.budgets)
    config.mask_budget = opt.mask_budget
    config.class_explainer_epochs = opt.class_explainer_epochs
    config.class_early_stop_delta = opt.class_early_stop_delta
    config.class_batch_size = opt.class_batch_size
    config.class_eval_batch_size = opt.class_eval_batch_size
    config.explainer_lr = opt.explainer_lr
    config.lime_samples = opt.lime_samples
    config.lime_alpha = opt.lime_alpha
    config.class_lime_max_samples = opt.class_lime_max_samples
    config.class_lime_perturbations = opt.class_lime_perturbations
    config.class_lime_use_conf_weight = opt.class_lime_use_conf_weight
    config.max_samples_per_class = opt.max_samples_per_class
    if opt.device is not None:
        config.device = opt.device
    elif not torch.cuda.is_available():
        config.device = 'cpu'

    seed_everything(config.SEED)
    model, header_model, pack_model, header_pack_model = load_models(config)
    unique_labels = class_labels(config)
    test_datasets = []
    for class_label in unique_labels:
        dataset = TrafficDatasetClass(config.dataset, flag='test', class_idx=class_label, PAD_TRUNC_DIGIT=config.PAD_TRUNC_DIGIT)
        if opt.max_samples_per_class is not None:
            dataset = Subset(dataset, range(min(opt.max_samples_per_class, len(dataset))))
        test_datasets.append(dataset)
    test_dataloaders = [DataLoader(dataset, batch_size=config.class_eval_batch_size, shuffle=False) for dataset in test_datasets]
    run(test_dataloaders, unique_labels, model, header_model, pack_model, header_pack_model, config)
