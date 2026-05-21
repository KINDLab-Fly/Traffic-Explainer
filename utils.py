import datetime
import random

import torch
import numpy as np

from config import *
from torch.nn.functional import softmax
import torch.nn.functional as F

config = Config()


def budget_constrain(mask_header, mask_payload, budget1, budget2):
    return F.relu(torch.sigmoid(mask_header).mean() - budget1) + F.relu(torch.sigmoid(mask_payload).mean() - budget2)    

def cal_class_entropy_class(p, pred_label):
    p = softmax(p, dim = 1)
    p = torch.log2(p)
    pred_label = pred_label.reshape(-1)
    selected_elements = p[torch.arange(p.size(0)), pred_label]

    return -selected_elements.mean()

def cal_class_entropy_instance(p, pred_label):
    p = softmax(p, dim = 1)
    p = p.squeeze(0)

    p = torch.log2(p)
    
    selected_elements = p[pred_label]

    return -selected_elements.mean()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_emb(model, pack_model, seq, mask, seq_byte_length, seq_byte_mask = None, baseline = None, output_attentions = False):
    """
    seq_emb = model.embed(seq.reshape(-1, 150)) # (batch, seq_len, d_model)
    seq_emb = model.encode(seq_emb, mask = mask.reshape(-1, 150)) # (batch, seq_len, d_model)
    
    seq_emb = seq_emb * (1 - mask).reshape(-1, 150).unsqueeze(-1)
    seq_emb = seq_emb.mean(dim = 1).reshape(batch_size, -1, seq_emb.shape[-1]).mean(dim = 1) # (batch, d_model)
    """

    seq_flat = seq.reshape(-1, seq_byte_length)
    if seq_byte_mask == None:
        seq_emb = model.embed(seq_flat) # (batch, seq_len, d_model)
    else:
        seq_emb = model.embed(seq_flat)
        if seq_byte_mask.shape == seq.shape or seq_byte_mask.numel() == seq.numel():
            # Most explanation masks are byte-value masks; this branch keeps
            # support for explicit per-position masks used by some ablations.
            mask_flat = seq_byte_mask.reshape(-1, seq_byte_length).unsqueeze(-1).float()
            seq_emb = seq_emb * mask_flat
        else:
            seq_emb = seq_emb * seq_byte_mask[seq_flat].unsqueeze(-1) # (batch, seq_len, d_model)
    
    if baseline == 'Byte-Transformer wo Byte-att':
        byte_mask = (1 - mask).reshape(-1, seq_byte_length).unsqueeze(-1)  # (batch * num_packets, seq_byte_length, 1)
        seq_emb = seq_emb * byte_mask
        
        # print(seq_emb.shape)
        seq_emb = seq_emb.sum(dim=1) / byte_mask.sum(dim=1).clamp(min=1e-9)  # (batch * num_packets, d_model)

        seq_emb = seq_emb.reshape(seq.shape[0], -1, seq_emb.shape[-1]) # (batch, packet_len, d_model)

        seq_emb = pack_model.encode(seq_emb, mask = None).mean(dim = 1)
    
    elif baseline == 'Byte-Transformer wo Byte-Pack-att':
        # Step 1: mask out padded bytes
        byte_mask = (1 - mask).reshape(-1, seq_byte_length).unsqueeze(-1)  # (batch * num_packets, seq_byte_length, 1)
        seq_emb = seq_emb * byte_mask

        # Step 2: mean-pool over bytes (within each packet)
        seq_emb = seq_emb.sum(dim=1) / byte_mask.sum(dim=1).clamp(min=1e-9)  # (batch * num_packets, d_model)

        # Step 3: reshape to (batch, packet_len, d_model)
        seq_emb = seq_emb.reshape(seq.shape[0], -1, seq_emb.shape[-1])  # (batch, packet_len, d_model)

        # Step 4: mean-pool over packets (no pack_model encoding)
        seq_emb = seq_emb.mean(dim=1)  # (batch, d_model)

    elif baseline == 'Byte-Transformer':
        if output_attentions:
            seq_emb, attentions = model.encode(seq_emb, mask = mask.reshape(-1, seq_byte_length), output_attentions = True) # (batch, seq_len, d_model)
        else:
            seq_emb = model.encode(seq_emb, mask = mask.reshape(-1, seq_byte_length)) # (batch, seq_len, d_model)
        seq_emb = seq_emb * (1 - mask).reshape(-1, seq_byte_length).unsqueeze(-1)

        seq_emb = seq_emb.mean(dim = 1)
        seq_emb = seq_emb.reshape(seq.shape[0], -1, seq_emb.shape[-1])
        # print(seq_emb.shape)
        seq_emb = pack_model.encode(seq_emb, mask = None).mean(dim = 1)
    
    elif baseline == 'ET-Bert':
        seq_emb = model.encode(seq_emb, mask = mask.reshape(-1, seq_byte_length)) # (batch, seq_len, d_model)
        seq_emb = seq_emb * (1 - mask).reshape(-1, seq_byte_length).unsqueeze(-1)

        seq_emb = seq_emb.mean(dim = 1)

        seq_emb = seq_emb.reshape(seq.shape[0], -1, seq_emb.shape[-1])

        seq_emb = seq_emb.mean(dim = 1)


    if output_attentions:
        return seq_emb, attentions

    return seq_emb


def compute_emb_lime(model, pack_model, seq, mask, seq_byte_length, seq_byte_mask = None, baseline = None):
    """
    seq_emb = model.embed(seq.reshape(-1, 150)) # (batch, seq_len, d_model)
    seq_emb = model.encode(seq_emb, mask = mask.reshape(-1, 150)) # (batch, seq_len, d_model)
    
    seq_emb = seq_emb * (1 - mask).reshape(-1, 150).unsqueeze(-1)
    seq_emb = seq_emb.mean(dim = 1).reshape(batch_size, -1, seq_emb.shape[-1]).mean(dim = 1) # (batch, d_model)
    """

    if seq_byte_mask == None:
        seq_emb = model.embed(seq.reshape(-1, seq_byte_length)) # (batch, seq_len, d_model)
    else:
        seq_flat = seq.reshape(-1, seq.shape[-1])  # shape: [batch_size * num_packets, seq_len]
        mask_flat = seq_byte_mask.reshape(-1, seq.shape[-1]).unsqueeze(-1).float()  # shape: [batch_size * num_packets, seq_len, 1]

        # print(9999, model.embed(seq_flat).shape, mask_flat.shape)
        seq_emb = model.embed(seq_flat) * mask_flat  # shape: [batch_size * num_packets, seq_len, d_model]

    if baseline == 'Byte-Transformer wo Byte-att':
        byte_mask = (1 - mask).reshape(-1, seq_byte_length).unsqueeze(-1)  # (batch * num_packets, seq_byte_length, 1)
        seq_emb = seq_emb * byte_mask
        
        # print(seq_emb.shape)
        seq_emb = seq_emb.sum(dim=1) / byte_mask.sum(dim=1).clamp(min=1e-9)  # (batch * num_packets, d_model)

        seq_emb = seq_emb.reshape(seq.shape[0], -1, seq_emb.shape[-1]) # (batch, packet_len, d_model)

        seq_emb = pack_model.encode(seq_emb, mask = None).mean(dim = 1)
    
    elif baseline == 'Byte-Transformer wo Byte-Pack-att':
        # Step 1: mask out padded bytes
        byte_mask = (1 - mask).reshape(-1, seq_byte_length).unsqueeze(-1)  # (batch * num_packets, seq_byte_length, 1)
        seq_emb = seq_emb * byte_mask

        # Step 2: mean-pool over bytes (within each packet)
        seq_emb = seq_emb.sum(dim=1) / byte_mask.sum(dim=1).clamp(min=1e-9)  # (batch * num_packets, d_model)

        # Step 3: reshape to (batch, packet_len, d_model)
        seq_emb = seq_emb.reshape(seq.shape[0], -1, seq_emb.shape[-1])  # (batch, packet_len, d_model)

        # Step 4: mean-pool over packets (no pack_model encoding)
        seq_emb = seq_emb.mean(dim=1)  # (batch, d_model)

    elif baseline == 'Byte-Transformer':
        seq_emb = model.encode(seq_emb, mask = mask.reshape(-1, seq_byte_length)) # (batch, seq_len, d_model)
        seq_emb = seq_emb * (1 - mask).reshape(-1, seq_byte_length).unsqueeze(-1)

        seq_emb = seq_emb.mean(dim = 1)
        seq_emb = seq_emb.reshape(seq.shape[0], -1, seq_emb.shape[-1])
        # print(seq_emb.shape)
        seq_emb = pack_model.encode(seq_emb, mask = None).mean(dim = 1)
    
    elif baseline == 'ET-Bert':
        seq_emb = model.encode(seq_emb, mask = mask.reshape(-1, seq_byte_length)) # (batch, seq_len, d_model)
        seq_emb = seq_emb * (1 - mask).reshape(-1, seq_byte_length).unsqueeze(-1)

        seq_emb = seq_emb.mean(dim = 1)

        seq_emb = seq_emb.reshape(seq.shape[0], -1, seq_emb.shape[-1])

        seq_emb = seq_emb.mean(dim = 1)


    return seq_emb



def get_device(index=3):
    return torch.device("cuda:" + str(index) if torch.cuda.is_available() else "cpu")


def show_time():
    time_stamp = '\033[1;31;40m[' + str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + ']\033[0m'

    return time_stamp


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # dgl.seed(seed)
    # dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_bytes_from_raw(s):
    rows = s.split('\n')
    for i, row in enumerate(rows):
        rows[i] = row[6: 53].strip()

    bytes_list = []
    for row in rows:
        bytes_list.extend(row.split(' '))

    bytes_list_dec = [int(hex, 16) for hex in bytes_list]

    return bytes_list, bytes_list_dec


def pad_truncate(flow, type, config):
    flow_pad_trunc_length = config.FLOW_PAD_TRUNC_LENGTH
    if type == 'payload':
        byte_pad_trunc_length = config.BYTE_PAD_TRUNC_LENGTH
    elif type == 'header':
        byte_pad_trunc_length = config.HEADER_BYTE_PAD_TRUNC_LENGTH

    if len(flow) > flow_pad_trunc_length:
        flow = flow[:flow_pad_trunc_length]

    for ind, p in enumerate(flow):
        if len(p) > byte_pad_trunc_length:
            flow[ind] = p[:byte_pad_trunc_length]
        elif len(p) < byte_pad_trunc_length:
            p.extend([config.PAD_TRUNC_DIGIT] * (byte_pad_trunc_length - len(p)))
            flow[ind] = p

    if len(flow) < flow_pad_trunc_length:
        flow.extend([[config.PAD_TRUNC_DIGIT] * byte_pad_trunc_length] * (flow_pad_trunc_length - len(flow)))

    return flow


def remove(flow):
    for ind, p in enumerate(flow):
        ip_header = p[:20]
        tcp_udp_header = p[20:]
        ip_header = ip_header[:12]
        tcp_udp_header = tcp_udp_header[4:]

        renew_header = []
        renew_header.extend(ip_header)
        renew_header.extend(tcp_udp_header)
        flow[ind] = renew_header

    return flow


def split_flow_ISCX(file_path, cate, allow_empty, pad_trunc, config, type='payload'):
    file = np.load(file_path, allow_pickle=True)
    packets = file[type]
    if type == 'header':
        baseline = file['payload']
    data_list = []

    seg_pcap = packets
    if type == 'header':
        seg_baseline = baseline
    if allow_empty:
        seg_pcap = [list(p) for ind, p in enumerate(seg_pcap)]
        if type == 'header':
            seg_baseline = [list(p) for ind, p in enumerate(seg_baseline)]
    else:
        seg_pcap = [list(p) for ind, p in enumerate(seg_pcap) if len(p) != 0]
        if type == 'header':
            seg_baseline =[list(p) for ind, p in enumerate(seg_baseline) if len(p) != 0]
    if type == 'header':
        if len(seg_baseline) == 0:
            print("Empty Flow Detected")
            return data_list
    else:
        if len(seg_pcap) == 0:
            print("Empty Flow Detected")
            return data_list
    if pad_trunc:
        if type == 'header':
            if len(seg_baseline) > config.ANOMALOUS_FLOW_THRESHOLD:
                print("Anomalous Flow Detected")
                return data_list
            seg_pcap = remove(flow=seg_pcap)
        else:
            if len(seg_pcap) > config.ANOMALOUS_FLOW_THRESHOLD:
                print("Anomalous Flow Detected")
                return data_list
        seg_pcap = pad_truncate(flow=seg_pcap, type=type, config=config)
    data_list.append(seg_pcap)

    return data_list


def split_flow_Tor_nonoverlapping(file_path, cate, allow_empty, pad_trunc, config, type='payload'):
    file = np.load(file_path, allow_pickle=True)
    packets = file[type]
    if type == 'header':
        baseline = file['payload']
    data_list = []
    time_stamp = np.array(file['time']).astype(np.float64)
    time_stamp = time_stamp - time_stamp[0]
    sliding_window = int((time_stamp[-1] - 60) / 60) + 1
    if time_stamp[-1] <= 60:
        sliding_window = 1
    begin = [60 * i for i in range(sliding_window)]
    end = [60 + 60 * i for i in range(sliding_window)]
    all_seg_stamp = list(set(begin + end))
    all_seg_stamp.sort()
    stamp_ind_map = dict()
    prev_j = 0
    for i, seg_stamp in enumerate(all_seg_stamp):
        for j in range(prev_j, len(time_stamp)):
            if seg_stamp <= time_stamp[j]:
                stamp_ind_map[seg_stamp] = j
                prev_j = j
                break
    if time_stamp[-1] <= 60:
        stamp_ind_map[60] = len(time_stamp)
    begin = [stamp_ind_map[i] for i in begin]
    end = [stamp_ind_map[i] for i in end]
    for s_ind, e_ind in zip(begin, end):
        if s_ind == e_ind:
            continue
        seg_pcap = packets[s_ind: e_ind]
        if type == 'header':
            seg_baseline = baseline[s_ind: e_ind]
        if allow_empty:
            seg_pcap = [list(p) for ind, p in enumerate(seg_pcap)]
            if type == 'header':
                seg_baseline = [list(p) for ind, p in enumerate(seg_baseline)]
        else:
            seg_pcap = [list(p) for ind, p in enumerate(seg_pcap) if len(p) != 0]
            if type == 'header':
                seg_baseline =[list(p) for ind, p in enumerate(seg_baseline) if len(p) != 0]
        if type == 'header':
            if len(seg_baseline) == 0:
                print("Empty Flow Detected")
                continue
        else:
            if len(seg_pcap) == 0:
                print("Empty Flow Detected")
                continue
        if pad_trunc:
            if type == 'header':
                if len(seg_baseline) > config.ANOMALOUS_FLOW_THRESHOLD:
                    print("Anomalous Flow Detected")
                    continue
                seg_pcap = remove(flow=seg_pcap)
            else:
                if len(seg_pcap) > config.ANOMALOUS_FLOW_THRESHOLD:
                    print("Anomalous Flow Detected")
                    continue
            seg_pcap = pad_truncate(flow=seg_pcap, type=type, config=config)
        data_list.append(seg_pcap)

    return data_list


def construct_graph(bytes, w_size, k=1):
    # word co-occurence with context windows
    window_size = w_size
    windows = [] # [[], [], [], ..., []]

    words = bytes # ['A', 'B', 'C']
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        # print(length, length - window_size + 1)
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)


    word_window_freq = {}
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])


    word_pair_count = {}
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_i_id = word_i
                word_j = window[j]
                word_j_id = word_j
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1

    src = []
    dst = []
    weight = []

    # pmi as weights

    num_window = len(windows)

    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[i]
        word_freq_j = word_window_freq[j]
        pmi = math.log((1.0 * count / num_window) ** k /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue
        src.append(i)
        dst.append(j)
        weight.append(pmi)

    bytes2id = {}
    feat = []
    id_count = 0
    for byte in src:
        if byte in bytes2id:
            continue
        bytes2id[byte] = id_count
        id_count += 1
        feat.append([byte])

    src = [bytes2id[i] for i in src]
    dst = [bytes2id[i] for i in dst]

    g = dgl.graph((src, dst))
    g.ndata['feat'] = torch.tensor(feat, dtype=torch.float32)

    return dgl.add_self_loop(g)



def construct_graph_pyg(bytes, w_size, k=1):
    # word co-occurence with context windows
    window_size = w_size
    windows = [] # [[], [], [], ..., []]

    words = bytes # ['A', 'B', 'C']
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        # print(length, length - window_size + 1)
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)


    word_window_freq = {}
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])


    word_pair_count = {}
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_i_id = word_i
                word_j = window[j]
                word_j_id = word_j
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1

    src = []
    dst = []
    weight = []

    # pmi as weights

    num_window = len(windows)

    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[i]
        word_freq_j = word_window_freq[j]
        pmi = math.log((1.0 * count / num_window) ** k /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue
        src.append(i)
        dst.append(j)
        weight.append(pmi)

    bytes2id = {}
    feat = []
    id_count = 0
    for byte in src:
        if byte in bytes2id:
            continue
        bytes2id[byte] = id_count
        id_count += 1
        feat.append([byte])

    src = [bytes2id[i] for i in src]
    dst = [bytes2id[i] for i in dst]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x = torch.tensor(feat, dtype=torch.float32)

    edge_index = to_undirected(edge_index, x.shape[0], reduce = 'mean')
    edge_index, _ = add_self_loops(edge_index)

    return Data(x = x, edge_index = edge_index)


if __name__ == '__main__':
    pass
