import argparse
import os
import subprocess
from collections import OrderedDict

import numpy as np


PAD_TOKEN = 256
FLOW_LEN = 50
PAYLOAD_LEN = 150
HEADER_LEN = 40
ANOMALOUS_FLOW_THRESHOLD = 10000
MAX_SEG_PER_CLASS = 9999


DATASETS = {
    'iscx-vpn': {
        'directory': 'ISCX-VPN-2016',
        'classes': OrderedDict([
            (0, ('Chat', ['chat'])),
            (1, ('Email', ['email'])),
            (2, ('File', ['ftps', 'sftp', 'files'])),
            (3, ('P2P', ['bittorrent'])),
            (4, ('Streaming', ['netflix', 'spotify', 'vimeo', 'youtube'])),
            (5, ('VoIP', ['audio', 'voipbuster'])),
        ]),
        'tor_windows': False,
    },
    'iscx-nonvpn': {
        'directory': 'ISCX-NonVPN-2016',
        'classes': OrderedDict([
            (0, ('Chat', ['chat'])),
            (1, ('Email', ['email'])),
            (2, ('File', ['ftps', 'sftp', 'file', 'scp'])),
            (3, ('Streaming', ['netflix', 'spotify', 'vimeo', 'youtube'])),
            (4, ('Video', ['video'])),
            (5, ('VoIP', ['voipbuster', 'audio'])),
        ]),
        'tor_windows': False,
    },
    'iscx-tor': {
        'directory': 'ISCX-Tor-2017',
        'classes': OrderedDict([
            (0, ('Audio-Streaming', ['AUDIO', 'spotify'])),
            (1, ('Browsing', ['BROWSING'])),
            (2, ('Chat', ['CHAT'])),
            (3, ('File', ['FILE'])),
            (4, ('Mail', ['MAIL'])),
            (5, ('P2P', ['P2P', 'p2p'])),
            (6, ('Video-Streaming', ['VIDEO'])),
            (7, ('VoIP', ['VOIP'])),
        ]),
        'tor_windows': True,
    },
    'iscx-nontor': {
        'directory': 'ISCX-NonTor-2017',
        'classes': OrderedDict([
            (0, ('Audio', ['Audio', 'spotify'])),
            (1, ('Browsing', ['browsing', 'Browsing', 'ssl'])),
            (2, ('Chat', ['chat', 'Chat'])),
            (3, ('Email', ['Thunderbird', 'POP', 'Email'])),
            (4, ('FTP', ['FTP', 'transfer'])),
            (5, ('P2P', ['p2p'])),
            (6, ('Video', ['Youtube', 'Vimeo'])),
            (7, ('VoIP', ['Voice', 'voice'])),
        ]),
        'tor_windows': False,
    },
}


def class_dir(dataset_dir, class_name):
    return os.path.join(dataset_dir, 'process_file', class_name)


def run_splitcap(dataset_key, dataset_root, overwrite):
    spec = DATASETS[dataset_key]
    dataset_dir = os.path.join(dataset_root, spec['directory'])
    raw_dir = os.path.join(dataset_dir, 'raw')
    splitcap = os.path.join(dataset_dir, 'SplitCap.exe')
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f'Missing raw pcap directory: {raw_dir}')
    if not os.path.exists(splitcap):
        raise FileNotFoundError(f'Missing SplitCap.exe: {splitcap}')

    for _, (class_name, patterns) in spec['classes'].items():
        os.makedirs(class_dir(dataset_dir, class_name), exist_ok=True)

    for filename in sorted(os.listdir(raw_dir)):
        input_path = os.path.join(raw_dir, filename)
        if not os.path.isfile(input_path):
            continue
        for _, (class_name, patterns) in spec['classes'].items():
            if any(pattern in filename for pattern in patterns):
                output_dir = class_dir(dataset_dir, class_name)
                if overwrite or not any(name.startswith(filename) and name.endswith('.pcap') for name in os.listdir(output_dir)):
                    subprocess.run(
                        ['mono', splitcap, '-r', input_path, '-s', 'session', '-o', output_dir],
                        check=True,
                    )
                remove_udp_sessions(output_dir)
                break


def remove_udp_sessions(output_dir):
    for filename in os.listdir(output_dir):
        if 'UDP' in filename and filename.endswith('.pcap'):
            os.remove(os.path.join(output_dir, filename))


def packet_bytes(pkt):
    return list(bytes(pkt))


def convert_pcap_to_npz(pcap_path, npz_path):
    try:
        from scapy.all import IP, Raw, TCP, rdpcap
    except ImportError as exc:
        raise ImportError('Install scapy to run --stage pcap2npz') from exc

    headers = []
    payloads = []
    payload_length = []
    pkt_length = []
    src_ip = []
    dst_ip = []
    src_port = []
    dst_port = []
    times = []
    protocol = []
    flags = []
    mss = []

    for pkt in rdpcap(pcap_path):
        if not pkt.haslayer(IP) or not pkt.haslayer(TCP):
            continue
        full_packet = packet_bytes(pkt)
        payload = packet_bytes(pkt[Raw].load) if pkt.haslayer(Raw) else []
        header = full_packet[: len(full_packet) - len(payload)]
        tcp = pkt[TCP]
        ip = pkt[IP]

        headers.append(header)
        payloads.append(payload)
        payload_length.append(len(payload))
        pkt_length.append(len(header) + len(payload))
        src_ip.append(ip.src)
        dst_ip.append(ip.dst)
        src_port.append(int(tcp.sport))
        dst_port.append(int(tcp.dport))
        times.append(float(pkt.time))
        protocol.append(int(ip.proto))
        flags.append(str(tcp.flags))
        mss.append(next((value for key, value in tcp.options if key == 'MSS'), 0))

    np.savez_compressed(
        npz_path,
        header=np.array(headers, dtype=object),
        payload=np.array(payloads, dtype=object),
        payload_length=np.array(payload_length, dtype=object),
        pkt_length=np.array(pkt_length, dtype=object),
        src_ip=np.array(src_ip, dtype=object),
        dst_ip=np.array(dst_ip, dtype=object),
        src_port=np.array(src_port, dtype=object),
        dst_port=np.array(dst_port, dtype=object),
        time=np.array(times, dtype=object),
        protocol=np.array(protocol, dtype=object),
        flag=np.array(flags, dtype=object),
        mss=np.array(mss, dtype=object),
    )


def run_pcap2npz(dataset_key, dataset_root, overwrite, file_order):
    spec = DATASETS[dataset_key]
    dataset_dir = os.path.join(dataset_root, spec['directory'])
    for _, (class_name, _) in spec['classes'].items():
        directory = class_dir(dataset_dir, class_name)
        filenames = list_files(directory, file_order)
        for filename in filenames:
            if not filename.endswith('.pcap'):
                continue
            pcap_path = os.path.join(directory, filename)
            npz_path = os.path.join(directory, filename[:-4] + 'npz')
            if os.path.exists(npz_path) and not overwrite:
                continue
            print(f'pcap2npz {pcap_path}')
            convert_pcap_to_npz(pcap_path, npz_path)


def list_files(directory, file_order):
    filenames = os.listdir(directory)
    if file_order == 'sorted':
        filenames = sorted(filenames)
    return filenames


def pad_truncate(flow, byte_len):
    flow = [list(packet) for packet in flow[:FLOW_LEN]]
    for index, packet in enumerate(flow):
        if len(packet) > byte_len:
            flow[index] = packet[:byte_len]
        else:
            flow[index] = packet + [PAD_TOKEN] * (byte_len - len(packet))
    while len(flow) < FLOW_LEN:
        flow.append([PAD_TOKEN] * byte_len)
    return flow


def remove_header_identifiers(flow):
    output = []
    for packet in flow:
        ip_header = packet[:20]
        tcp_udp_header = packet[20:]
        output.append(ip_header[:12] + tcp_udp_header[4:])
    return output


def split_flow(file_path, is_tor, data_type):
    loaded = np.load(file_path, allow_pickle=True)
    packets = loaded[data_type]
    baseline = loaded['payload'] if data_type == 'header' else packets
    if len(baseline) == 0:
        return []

    if is_tor:
        return split_tor_windows(loaded, data_type)
    segment = prepare_segment(packets, baseline, data_type)
    return [segment] if segment else []


def split_tor_windows(loaded, data_type):
    packets = loaded[data_type]
    baseline = loaded['payload'] if data_type == 'header' else packets
    timestamps = np.asarray(loaded['time'], dtype=np.float64)
    if len(timestamps) == 0:
        return []
    timestamps = timestamps - timestamps[0]
    windows = int((timestamps[-1] - 60) / 60) + 1
    if timestamps[-1] <= 60:
        windows = 1

    outputs = []
    for window_index in range(windows):
        start_time = 60 * window_index
        end_time = start_time + 60
        start = int(np.searchsorted(timestamps, start_time, side='left'))
        end = int(np.searchsorted(timestamps, end_time, side='left'))
        if timestamps[-1] <= 60:
            end = len(timestamps)
        if start == end:
            continue
        segment = prepare_segment(packets[start:end], baseline[start:end], data_type)
        if segment:
            outputs.append(segment)
    return outputs


def prepare_segment(packets, baseline, data_type):
    packets = [list(packet) for packet in packets if len(packet) != 0]
    baseline = [list(packet) for packet in baseline if len(packet) != 0]
    if not baseline:
        return []
    if len(baseline) > ANOMALOUS_FLOW_THRESHOLD:
        return []
    if data_type == 'header':
        packets = remove_header_identifiers(packets)
        return pad_truncate(packets, HEADER_LEN)
    return pad_truncate(packets, PAYLOAD_LEN)


def output_paths(dataset_dir, data_type):
    prefix = 'header_' if data_type == 'header' else ''
    return {
        'train': os.path.join(dataset_dir, f'{prefix}train_pyg.npz'),
        'val': os.path.join(dataset_dir, f'{prefix}val_pyg.npz'),
        'test': os.path.join(dataset_dir, f'{prefix}test_pyg.npz'),
    }


def build_arrays(dataset_key, dataset_root, file_order, overwrite):
    spec = DATASETS[dataset_key]
    dataset_dir = os.path.join(dataset_root, spec['directory'])
    for data_type in ('payload', 'header'):
        paths = output_paths(dataset_dir, data_type)
        if not overwrite and any(os.path.exists(path) for path in paths.values()):
            raise FileExistsError(f'Output arrays already exist for {dataset_key}/{data_type}; use --overwrite')

        splits = {'train': [], 'val': [], 'test': []}
        labels = {'train': [], 'val': [], 'test': []}
        counts = {}

        for class_id, (class_name, _) in spec['classes'].items():
            directory = class_dir(dataset_dir, class_name)
            data_list = []
            for filename in list_files(directory, file_order):
                if not filename.endswith('.npz'):
                    continue
                file_path = os.path.join(directory, filename)
                data_list.extend(split_flow(file_path, spec['tor_windows'], data_type))

            data_list = data_list[:MAX_SEG_PER_CLASS]
            train_end = int(len(data_list) * 0.8)
            val_end = int(len(data_list) * 0.9)
            split_data = {
                'train': data_list[:train_end],
                'val': data_list[train_end:val_end],
                'test': data_list[val_end:],
            }
            for split_name, values in split_data.items():
                splits[split_name].extend(values)
                labels[split_name].extend([class_id] * len(values))
            counts[class_id] = {name: len(values) for name, values in split_data.items()}

        for split_name, path in paths.items():
            np.savez_compressed(
                path,
                data=np.asarray(splits[split_name], dtype=np.int64),
                label=np.asarray(labels[split_name], dtype=np.int64),
            )
        print(f'{dataset_key}/{data_type}: {counts}')


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess ISCX datasets for Traffic-Explainer.')
    parser.add_argument('--dataset', choices=sorted(DATASETS), required=True)
    parser.add_argument('--root', default='dataset')
    parser.add_argument(
        '--stage',
        choices=['splitcap', 'pcap2npz', 'build', 'all'],
        default='build',
        help='build creates the runtime train/val/test npz arrays from per-session npz files.',
    )
    parser.add_argument('--file_order', choices=['filesystem', 'sorted'], default='filesystem')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.stage in {'splitcap', 'all'}:
        run_splitcap(args.dataset, args.root, args.overwrite)
    if args.stage in {'pcap2npz', 'all'}:
        run_pcap2npz(args.dataset, args.root, args.overwrite, args.file_order)
    if args.stage in {'build', 'all'}:
        build_arrays(args.dataset, args.root, args.file_order, args.overwrite)
