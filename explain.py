import argparse
import os
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.linear_model import Ridge

try:
    import shap
except ImportError:
    shap = None

from config import get_config
from dataset import TrafficDataset
from model import build_transformer, explainer_model
from utils import budget_constrain, cal_class_entropy_instance, compute_emb, seed_everything


EXPLANATION_ALIASES = {
    'traffic-explainer': 'byte-level',
    'byte-level': 'byte-level',
    'random': 'random',
    'saliency': 'saliency map',
    'saliency-map': 'saliency map',
    'saliency map': 'saliency map',
    'lime': 'lime',
    'shap': 'shap',
    'att-rollout': 'att-rollout',
    'att_rollout': 'att-rollout',
    'attention-rollout': 'att-rollout',
}


def parse_budgets(value):
    return [float(item.strip()) for item in value.split(',') if item.strip()]


def predict_logits(model, header_model, pack_model, header_pack_model, seq, mask, seq_header, seq_header_mask, config, payload_mask=None, header_mask=None):
    seq_emb = compute_emb(
        model,
        pack_model,
        seq,
        mask,
        config.BYTE_PAD_TRUNC_LENGTH,
        seq_byte_mask=payload_mask,
        baseline=config.baseline,
    )
    header_emb = compute_emb(
        header_model,
        header_pack_model,
        seq_header,
        seq_header_mask,
        config.HEADER_BYTE_PAD_TRUNC_LENGTH,
        seq_byte_mask=header_mask,
        baseline=config.baseline,
    )
    return model.project(torch.cat([seq_emb, header_emb], dim=1))


def is_localization_config(config):
    return getattr(config, 'data_format', None) == 'pkl'


def explanation_byte_ids(config, device=None):
    if is_localization_config(config):
        return torch.arange(256, dtype=torch.long, device=device)
    return torch.arange(config.seq_vocab_size, dtype=torch.long, device=device)


def topk_from_scores(scores, budget, config=None):
    if config is None:
        candidate_ids = torch.arange(scores.shape[0], dtype=torch.long, device=scores.device)
    else:
        candidate_ids = explanation_byte_ids(config, scores.device)
    k = max(1, int(candidate_ids.numel() * budget))
    top_local = torch.topk(scores[candidate_ids], min(k, candidate_ids.numel())).indices
    return candidate_ids[top_local]


def random_byte_ids(config, budget):
    candidate_ids = explanation_byte_ids(config, config.device)
    k = max(1, int(candidate_ids.numel() * budget))
    return candidate_ids[torch.randperm(candidate_ids.numel(), device=config.device)[:k]]


def explanation_scores(scores, config):
    return scores[explanation_byte_ids(config, scores.device)]


def byte_value_scores_from_position_scores(seq, position_scores, seq_byte_length, vocab_size):
    scores = torch.zeros(vocab_size, dtype=torch.float32, device=seq.device)
    flat_seq = seq.reshape(-1, seq_byte_length)
    flat_scores = position_scores.reshape(-1, seq_byte_length)
    scores.scatter_add_(0, flat_seq.reshape(-1), flat_scores.reshape(-1))
    return scores


def lime_scores(model, header_model, pack_model, header_pack_model, seq, mask, seq_header, seq_header_mask, pred_label, config):
    candidate_ids = explanation_byte_ids(config, config.device)
    payload_masks = []
    header_masks = []
    probs = []

    with torch.no_grad():
        for _ in range(config.lime_samples):
            payload_mask_np = np.random.randint(0, 2, (candidate_ids.numel(),))
            header_mask_np = np.random.randint(0, 2, (candidate_ids.numel(),))
            payload_mask = torch.ones(config.seq_vocab_size, dtype=torch.float32, device=config.device)
            header_mask = torch.ones(config.seq_vocab_size, dtype=torch.float32, device=config.device)
            payload_mask[candidate_ids] = torch.tensor(payload_mask_np, dtype=torch.float32, device=config.device)
            header_mask[candidate_ids] = torch.tensor(header_mask_np, dtype=torch.float32, device=config.device)
            pred = predict_logits(
                model,
                header_model,
                pack_model,
                header_pack_model,
                seq,
                mask,
                seq_header,
                seq_header_mask,
                config,
                payload_mask=payload_mask,
                header_mask=header_mask,
            )
            probs.append(softmax(pred, dim=1)[0, int(pred_label[0])].item())
            payload_masks.append(payload_mask_np)
            header_masks.append(header_mask_np)

    payload_reg = Ridge(alpha=config.lime_alpha, fit_intercept=True)
    header_reg = Ridge(alpha=config.lime_alpha, fit_intercept=True)
    payload_reg.fit(np.array(payload_masks), np.array(probs))
    header_reg.fit(np.array(header_masks), np.array(probs))
    payload_scores = torch.zeros(config.seq_vocab_size, dtype=torch.float32, device=config.device)
    header_scores = torch.zeros(config.seq_vocab_size, dtype=torch.float32, device=config.device)
    payload_scores[candidate_ids] = torch.tensor(payload_reg.coef_, dtype=torch.float32, device=config.device)
    header_scores[candidate_ids] = torch.tensor(header_reg.coef_, dtype=torch.float32, device=config.device)
    return payload_scores, header_scores


def shap_scores(model, header_model, pack_model, header_pack_model, seq, mask, seq_header, seq_header_mask, pred_label, config):
    if shap is None:
        raise ImportError('The shap package is required for --explanation shap.')

    candidate_ids = explanation_byte_ids(config, config.device)

    def predict(byte_mask_batch):
        preds = []
        with torch.no_grad():
            for byte_mask in byte_mask_batch:
                byte_mask_tensor = torch.tensor(byte_mask, dtype=torch.float32, device=config.device)
                payload_mask = torch.ones(config.seq_vocab_size, dtype=torch.float32, device=config.device)
                header_mask = torch.ones(config.seq_vocab_size, dtype=torch.float32, device=config.device)
                payload_mask[candidate_ids] = byte_mask_tensor[:candidate_ids.numel()]
                header_mask[candidate_ids] = byte_mask_tensor[candidate_ids.numel():]
                pred = predict_logits(
                    model,
                    header_model,
                    pack_model,
                    header_pack_model,
                    seq,
                    mask,
                    seq_header,
                    seq_header_mask,
                    config,
                    payload_mask=payload_mask,
                    header_mask=header_mask,
                )
                preds.append(softmax(pred, dim=1)[0, int(pred_label[0])].detach().cpu().item())
        return np.array(preds)

    feature_size = candidate_ids.numel() * 2
    background = np.zeros((1, feature_size), dtype=np.float32)
    input_mask = np.ones((1, feature_size), dtype=np.float32)
    explainer = shap.KernelExplainer(predict, background)
    values = explainer.shap_values(input_mask, nsamples=config.shap_samples)
    values = np.asarray(values)
    values = values[0] if values.ndim == 2 else values.reshape(-1)
    payload_scores = torch.zeros(config.seq_vocab_size, dtype=torch.float32, device=config.device)
    header_scores = torch.zeros(config.seq_vocab_size, dtype=torch.float32, device=config.device)
    payload_scores[candidate_ids] = torch.tensor(values[:candidate_ids.numel()], dtype=torch.float32, device=config.device)
    header_scores[candidate_ids] = torch.tensor(values[candidate_ids.numel():], dtype=torch.float32, device=config.device)
    return payload_scores, header_scores


def rollout_scores_for_sequence(attentions, seq, seq_byte_length, vocab_size):
    if not attentions:
        return torch.zeros(vocab_size, dtype=torch.float32, device=seq.device)

    token_scores = []
    for sample_idx in range(seq.reshape(-1, seq_byte_length).shape[0]):
        rollout = None
        for attention in attentions:
            attention = attention[sample_idx].mean(dim=0)
            identity = torch.eye(attention.shape[0], device=attention.device)
            attention = 0.5 * attention + 0.5 * identity
            rollout = attention if rollout is None else attention @ rollout
        token_scores.append(rollout[0])

    position_scores = torch.stack(token_scores, dim=0)
    return byte_value_scores_from_position_scores(seq, position_scores, seq_byte_length, vocab_size)


def attention_rollout_scores(model, header_model, pack_model, header_pack_model, seq, mask, seq_header, seq_header_mask, config):
    if config.baseline != 'Byte-Transformer':
        raise ValueError('ATT Rollout is currently implemented for Byte-Transformer.')

    with torch.no_grad():
        _, payload_attentions = compute_emb(
            model,
            pack_model,
            seq,
            mask,
            config.BYTE_PAD_TRUNC_LENGTH,
            seq_byte_mask=None,
            baseline=config.baseline,
            output_attentions=True,
        )
        _, header_attentions = compute_emb(
            header_model,
            header_pack_model,
            seq_header,
            seq_header_mask,
            config.HEADER_BYTE_PAD_TRUNC_LENGTH,
            seq_byte_mask=None,
            baseline=config.baseline,
            output_attentions=True,
        )

    payload_scores = rollout_scores_for_sequence(payload_attentions, seq, config.BYTE_PAD_TRUNC_LENGTH, config.seq_vocab_size)
    header_scores = rollout_scores_for_sequence(header_attentions, seq_header, config.HEADER_BYTE_PAD_TRUNC_LENGTH, config.seq_vocab_size)
    return payload_scores, header_scores


def evaluate_selected_bytes(model, header_model, pack_model, header_pack_model, seq, mask, seq_header, seq_header_mask, top_payload, top_header, config):
    header_mask = torch.ones(config.seq_vocab_size, dtype=torch.float32, device=config.device)
    payload_mask = torch.ones(config.seq_vocab_size, dtype=torch.float32, device=config.device)
    header_mask[top_header] = 0
    payload_mask[top_payload] = 0

    rm_logits = predict_logits(
        model,
        header_model,
        pack_model,
        header_pack_model,
        seq,
        mask,
        seq_header,
        seq_header_mask,
        config,
        payload_mask=payload_mask,
        header_mask=header_mask,
    )
    pred_rm = rm_logits.argmax(dim=1).detach().cpu().item()

    add_logits = predict_logits(
        model,
        header_model,
        pack_model,
        header_pack_model,
        seq,
        mask,
        seq_header,
        seq_header_mask,
        config,
        payload_mask=1 - payload_mask,
        header_mask=1 - header_mask,
    )
    pred_add = add_logits.argmax(dim=1).detach().cpu().item()
    return pred_rm, pred_add


def compute_metrics(ground_truths, pred_labels, pred_rm_labels, pred_add_labels):
    labels = torch.tensor(ground_truths)
    preds = torch.tensor(pred_labels)
    rows = {}
    for budget in sorted(pred_add_labels.keys()):
        add = torch.tensor(pred_add_labels[budget]).view(-1)
        rm = torch.tensor(pred_rm_labels[budget]).view(-1)
        rows[budget] = {
            'Fid': (add == preds).float().mean().item(),
            'Acc': (add == labels).float().mean().item(),
            'C-Fid': (rm != preds).float().mean().item(),
            'C-Acc': (rm != labels).float().mean().item(),
        }
    return rows


def print_metrics(metrics):
    print('Budget\tFid\tAcc\tC-Fid\tC-Acc')
    for budget, row in metrics.items():
        print(
            f'{budget:.2f}\t'
            f'{row["Fid"] * 100:.2f}\t'
            f'{row["Acc"] * 100:.2f}\t'
            f'{row["C-Fid"] * 100:.2f}\t'
            f'{row["C-Acc"] * 100:.2f}'
        )


def run(test_dataloader, model, header_model, pack_model, header_pack_model, config):
    model.eval()
    header_model.eval()
    pack_model.eval()
    header_pack_model.eval()
    if config.freeze_predictor:
        for current_model in (model, header_model, pack_model, header_pack_model):
            for param in current_model.parameters():
                param.requires_grad_(False)

    ground_truths = []
    pred_labels = []
    pred_rm_labels = defaultdict(list)
    pred_add_labels = defaultdict(list)
    length_pad = []
    length_head = []
    explain_header_idx = defaultdict(list)
    explain_payload_idx = defaultdict(list)
    selected_header_idx = defaultdict(list)
    selected_payload_idx = defaultdict(list)

    for batch in tqdm(test_dataloader):
        seq, label, mask, seq_header, seq_header_mask = batch
        seq = seq.to(config.device, non_blocking=True)
        label = label.to(config.device, non_blocking=True)
        mask = mask.to(config.device, non_blocking=True)
        seq_header = seq_header.to(config.device, non_blocking=True)
        seq_header_mask = seq_header_mask.to(config.device, non_blocking=True)

        logits = predict_logits(model, header_model, pack_model, header_pack_model, seq, mask, seq_header, seq_header_mask, config)
        pred_label = softmax(logits, dim=1).argmax(dim=1).detach().cpu().numpy()

        ground_truths.extend(label.detach().cpu().numpy())
        pred_labels.extend(pred_label)
        length_pad.append(mask.sum().detach().cpu())
        length_head.append(seq_header_mask.sum().detach().cpu())

        if config.explain_strategy == 'byte-level':
            explainer = explainer_model(config).to(config.device)
            explainer.train()
            optimizer = optim.Adam(explainer.parameters(), lr=config.explainer_lr)

            for _ in range(config.explainer_epochs):
                pred = explainer(model, header_model, pack_model, header_pack_model, seq, mask, seq_header, seq_header_mask)
                loss_explain = cal_class_entropy_instance(pred, pred_label)
                loss_budget = budget_constrain(
                    explanation_scores(explainer.mask_header, config),
                    explanation_scores(explainer.mask_payload, config),
                    config.mask_budget,
                    config.mask_budget,
                )
                loss = loss_explain + loss_budget
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            payload_scores = torch.sigmoid(explainer.mask_payload).detach()
            header_scores = torch.sigmoid(explainer.mask_header).detach()

        elif config.explain_strategy == 'saliency map':
            if not config.legacy_saliency_accumulation:
                model.zero_grad(set_to_none=True)
                header_model.zero_grad(set_to_none=True)
                pack_model.zero_grad(set_to_none=True)
                header_pack_model.zero_grad(set_to_none=True)
            logits.squeeze(0)[int(pred_label[0])].backward()
            grad_payload = torch.norm(model.seq_embed.embedding.weight.grad, dim=1)
            grad_header = torch.norm(header_model.seq_embed.embedding.weight.grad, dim=1)
            shared_scores = 0.5 * (grad_payload + grad_header)
            payload_scores = shared_scores.detach()
            header_scores = shared_scores.detach()

        elif config.explain_strategy == 'lime':
            payload_scores, header_scores = lime_scores(
                model,
                header_model,
                pack_model,
                header_pack_model,
                seq,
                mask,
                seq_header,
                seq_header_mask,
                pred_label,
                config,
            )

        elif config.explain_strategy == 'shap':
            payload_scores, header_scores = shap_scores(
                model,
                header_model,
                pack_model,
                header_pack_model,
                seq,
                mask,
                seq_header,
                seq_header_mask,
                pred_label,
                config,
            )

        elif config.explain_strategy == 'att-rollout':
            payload_scores, header_scores = attention_rollout_scores(
                model,
                header_model,
                pack_model,
                header_pack_model,
                seq,
                mask,
                seq_header,
                seq_header_mask,
                config,
            )

        for budget in config.eval_budgets:
            if config.explain_strategy == 'random':
                top_header = random_byte_ids(config, budget)
                top_payload = random_byte_ids(config, budget)
            else:
                top_header = topk_from_scores(header_scores, budget, config)
                top_payload = topk_from_scores(payload_scores, budget, config)

            pred_rm, pred_add = evaluate_selected_bytes(
                model,
                header_model,
                pack_model,
                header_pack_model,
                seq,
                mask,
                seq_header,
                seq_header_mask,
                top_payload,
                top_header,
                config,
            )
            pred_rm_labels[budget].append(pred_rm)
            pred_add_labels[budget].append(pred_add)

            if config.explain_strategy in {'byte-level', 'saliency map', 'lime', 'shap', 'att-rollout'}:
                selected_header_idx[budget].append(top_header.detach().cpu())
                selected_payload_idx[budget].append(top_payload.detach().cpu())

    out_dir = f'./res/{config.baseline}/{config.dataset}/{config.explain_strategy}'
    os.makedirs(out_dir, exist_ok=True)
    suffix = str(config.mask_budget)
    if getattr(config, 'max_samples', None) is not None:
        suffix = f'{suffix}_max{config.max_samples}'
    torch.save(ground_truths, f'{out_dir}/ground_truths_{suffix}.pt')
    torch.save(pred_labels, f'{out_dir}/pred_labels_{suffix}.pt')
    torch.save(dict(pred_rm_labels), f'{out_dir}/pred_rm_labels_{suffix}.pt')
    torch.save(dict(pred_add_labels), f'{out_dir}/pred_add_labels_{suffix}.pt')
    torch.save(length_pad, f'{out_dir}/length_pad_{suffix}.pt')
    torch.save(length_head, f'{out_dir}/length_head_{suffix}.pt')

    if config.explain_strategy == 'byte-level':
        torch.save(explain_header_idx, f'./res/{config.baseline}/{config.dataset}/explain_header_idx_{suffix}.pt')
        torch.save(explain_payload_idx, f'./res/{config.baseline}/{config.dataset}/explain_payload_idx_{suffix}.pt')

    if config.explain_strategy in {'byte-level', 'saliency map', 'lime', 'shap', 'att-rollout'}:
        torch.save(dict(selected_header_idx), f'{out_dir}/selected_header_idx_{suffix}.pt')
        torch.save(dict(selected_payload_idx), f'{out_dir}/selected_payload_idx_{suffix}.pt')

    metrics = compute_metrics(ground_truths, pred_labels, pred_rm_labels, pred_add_labels)
    print_metrics(metrics)
    return metrics


def load_models(config):
    header_model, header_pack_model = build_transformer(
        config.seq_vocab_size,
        config.seq_seq_len,
        config.seq_pack_len,
        config.class_size,
        header=True,
    )
    model, pack_model = build_transformer(
        config.seq_vocab_size,
        config.seq_seq_len,
        config.seq_pack_len,
        config.class_size,
    )

    map_location = torch.device(config.device)
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

    def load_state(path):
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=map_location)

    model.load_state_dict(load_state(f'{checkpoint_dir}/model.pth'))
    pack_model.load_state_dict(load_state(f'{checkpoint_dir}/model_pack.pth'))
    header_model.load_state_dict(load_state(f'{checkpoint_dir}/model_header.pth'))
    header_pack_model.load_state_dict(load_state(f'{checkpoint_dir}/model_header_pack.pth'))

    return (
        model.to(config.device),
        header_model.to(config.device),
        pack_model.to(config.device),
        header_pack_model.to(config.device),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['iscx-vpn', 'iscx-nonvpn', 'iscx-tor', 'iscx-nontor', 'ios', 'android', 'ios-cross-plat', 'android-cross-plat'], required=True)
    parser.add_argument('--baseline', type=str, default='Byte-Transformer', choices=['Byte-Transformer', 'Byte-Transformer wo Byte-att', 'Byte-Transformer wo Byte-Pack-att', 'ET-Bert'])
    parser.add_argument('--explanation', type=str, default='traffic-explainer', choices=list(EXPLANATION_ALIASES.keys()))
    parser.add_argument('--mask_budget', type=float, default=0.1)
    parser.add_argument('--budgets', type=str, default='0.01,0.05,0.1')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--explainer_epochs', type=int, default=200)
    parser.add_argument('--explainer_lr', type=float, default=0.01)
    parser.add_argument('--lime_samples', type=int, default=30)
    parser.add_argument('--lime_alpha', type=float, default=0.1)
    parser.add_argument('--shap_samples', type=int, default=1000)
    parser.add_argument('--freeze_predictor', action='store_true', help='Freeze predictor parameters during Traffic-Explainer optimization. Off by default to reproduce the original implementation.')
    parser.add_argument('--legacy_saliency_accumulation', action='store_true', help='Reproduce the original cumulative-gradient saliency implementation. By default, saliency clears gradients per instance.')
    parser.add_argument('--max_samples', type=int, default=None)
    opt = parser.parse_args()

    config = get_config(opt.dataset)
    config.baseline = opt.baseline
    config.explain_strategy = EXPLANATION_ALIASES[opt.explanation]
    config.mask_budget = opt.mask_budget
    config.eval_budgets = parse_budgets(opt.budgets)
    config.explainer_epochs = opt.explainer_epochs
    config.explainer_lr = opt.explainer_lr
    config.lime_samples = opt.lime_samples
    config.lime_alpha = opt.lime_alpha
    config.shap_samples = opt.shap_samples
    config.freeze_predictor = opt.freeze_predictor
    config.legacy_saliency_accumulation = opt.legacy_saliency_accumulation
    config.max_samples = opt.max_samples
    if opt.device is not None:
        config.device = opt.device
    elif not torch.cuda.is_available():
        config.device = 'cpu'

    seed_everything(config.SEED)
    model, header_model, pack_model, header_pack_model = load_models(config)

    test_dataset = TrafficDataset(config.dataset, flag='test', PAD_TRUNC_DIGIT=config.PAD_TRUNC_DIGIT)
    if opt.max_samples is not None:
        test_dataset = Subset(test_dataset, range(min(opt.max_samples, len(test_dataset))))
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    run(test_dataloader, model, header_model, pack_model, header_pack_model, config)
