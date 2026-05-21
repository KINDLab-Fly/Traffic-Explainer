'''
Training Configuration
'''
class Config:
    SEED = 32

    PAD_TRUNC_DIGIT = 256
    FLOW_PAD_TRUNC_LENGTH = 50
    BYTE_PAD_TRUNC_LENGTH = 150
    HEADER_BYTE_PAD_TRUNC_LENGTH = 40
    ANOMALOUS_FLOW_THRESHOLD = 10000


    seq_vocab_size = 257
    seq_seq_len = 150
    seq_pack_len = 50
    
    device = 'cuda:0'
    lr = 1e-3
    n_epochs = 100
    batch_size = 64

    PMI_WINDOW_SIZE = 5


DATASET_ALIASES = {
    'iscx-vpn': 'ISCX-VPN-2016',
    'iscx-nonvpn': 'ISCX-NonVPN-2016',
    'iscx-tor': 'ISCX-Tor-2017',
    'iscx-nontor': 'ISCX-NonTor-2017',
    'ios': 'IOS_Cross_Plat',
    'android': 'Android_Cross_Plat',
    'ios-cross-plat': 'IOS_Cross_Plat',
    'android-cross-plat': 'Android_Cross_Plat',
}


def get_config(dataset_name):
    if dataset_name == 'iscx-vpn':
        return ISCXVPNConfig()
    if dataset_name == 'iscx-nonvpn':
        return ISCXNonVPNConfig()
    if dataset_name == 'iscx-tor':
        return ISCXTorConfig()
    if dataset_name == 'iscx-nontor':
        return ISCXNonTorConfig()
    if dataset_name in {'ios', 'ios-cross-plat'}:
        return IOSCrossPlatConfig()
    if dataset_name in {'android', 'android-cross-plat'}:
        return AndroidCrossPlatConfig()
    raise ValueError(f'Unknown dataset: {dataset_name}')


class LocalizationConfig(Config):
    PAD_TRUNC_DIGIT = 65536
    BYTE_PAD_TRUNC_LENGTH = 64
    HEADER_BYTE_PAD_TRUNC_LENGTH = 64
    seq_vocab_size = 65537
    seq_seq_len = 100
    seq_pack_len = 100
    class_size = 3
    NUM_CLASSES = 3
    batch_size = 4096
    data_format = 'pkl'


class IOSCrossPlatConfig(LocalizationConfig):
    dataset = 'IOS_Cross_Plat'
    TRAIN_DATA = r'./dataset_localization/IOS_Cross_Plat/train.pkl'
    VAL_DATA = r'./dataset_localization/IOS_Cross_Plat/val.pkl'
    TEST_DATA = r'./dataset_localization/IOS_Cross_Plat/test.pkl'


class AndroidCrossPlatConfig(LocalizationConfig):
    dataset = 'Android_Cross_Plat'
    TRAIN_DATA = r'./dataset_localization/Android_Cross_Plat/train.pkl'
    VAL_DATA = r'./dataset_localization/Android_Cross_Plat/val.pkl'
    TEST_DATA = r'./dataset_localization/Android_Cross_Plat/test.pkl'


'''
ISCX-VPN Dataset Configuration
'''
class ISCXVPNConfig(Config):
    dataset = 'ISCX-VPN-2016'
    TRAIN_DATA = r'./dataset/ISCX-VPN-2016/train_pyg.npz'
    HEADER_TRAIN_DATA = r'./dataset/ISCX-VPN-2016/header_train_pyg.npz'
    VAL_DATA = r'./dataset/ISCX-VPN-2016/val_pyg.npz'
    HEADER_VAL_DATA = r'./dataset/ISCX-VPN-2016/header_val_pyg.npz'
    TEST_DATA = r'./dataset/ISCX-VPN-2016/test_pyg.npz'
    HEADER_TEST_DATA = r'./dataset/ISCX-VPN-2016/header_test_pyg.npz'

    TRAIN_GRAPH_DATA = r'./dataset/ISCX-VPN-2016/train_graph_pyg.pt'
    HEADER_TRAIN_GRAPH_DATA = r'./dataset/ISCX-VPN-2016/header_train_graph_pyg.pt'
    VAL_GRAPH_DATA = r'./dataset/ISCX-VPN-2016/val_graph_pyg.pt'
    HEADER_VAL_GRAPH_DATA = r'./dataset/ISCX-VPN-2016/header_val_graph_pyg.pt'
    TEST_GRAPH_DATA = r'./dataset/ISCX-VPN-2016/test_graph_pyg.pt'
    HEADER_TEST_GRAPH_DATA = r'./dataset/ISCX-VPN-2016/header_test_graph_pyg.pt'


    MIX_MODEL_CHECKPOINT = r'./checkpoints/mix_model_iscx_vpn.pth'
    
    class_size = 6
    NUM_CLASSES = 6
    MAX_SEG_PER_CLASS = 9999
    NUM_WORKERS = 5

    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION = 1
    MAX_EPOCH = 20
    LR = 1e-2
    LR_MIN = 1e-4
    LABEL_SMOOTHING = 0
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.4
    DOWNSTREAM_DROPOUT = 0.0
    EMBEDDING_SIZE = 64
    H_FEATS = 128

    DIR_PATH_DICT = {0: r'./dataset/ISCX-VPN-2016/process_file/Chat',
                     1: r'./dataset/ISCX-VPN-2016/process_file/Email',
                     2: r'./dataset/ISCX-VPN-2016/process_file/File',
                     3: r'./dataset/ISCX-VPN-2016/process_file/P2P',
                     4: r'./dataset/ISCX-VPN-2016/process_file/Streaming',
                     5: r'./dataset/ISCX-VPN-2016/process_file/VoIP',
                     }


'''
ISCX-NonVPN Dataset Configuration
'''
class ISCXNonVPNConfig(Config):
    dataset = 'ISCX-NonVPN-2016'
    TRAIN_DATA = r'./dataset/ISCX-NonVPN-2016/train_pyg.npz'
    HEADER_TRAIN_DATA = r'./dataset/ISCX-NonVPN-2016/header_train_pyg.npz'
    VAL_DATA = r'./dataset/ISCX-NonVPN-2016/val_pyg.npz'
    HEADER_VAL_DATA = r'./dataset/ISCX-NonVPN-2016/header_val_pyg.npz'
    TEST_DATA = r'./dataset/ISCX-NonVPN-2016/test_pyg.npz'
    HEADER_TEST_DATA = r'./dataset/ISCX-NonVPN-2016/header_test_pyg.npz'

    TRAIN_GRAPH_DATA = r'./dataset/ISCX-NonVPN-2016/train_graph_pyg.pt'
    HEADER_TRAIN_GRAPH_DATA = r'./dataset/ISCX-NonVPN-2016/header_train_graph_pyg.pt'
    VAL_GRAPH_DATA = r'./dataset/ISCX-NonVPN-2016/val_graph_pyg.pt'
    HEADER_VAL_GRAPH_DATA = r'./dataset/ISCX-NonVPN-2016/header_val_graph_pyg.pt'
    TEST_GRAPH_DATA = r'./dataset/ISCX-NonVPN-2016/test_graph_pyg.pt'
    HEADER_TEST_GRAPH_DATA = r'./dataset/ISCX-NonVPN-2016/header_test_graph_pyg.pt'


    MIX_MODEL_CHECKPOINT = r'./checkpoints/mix_model_iscx_nonvpn.pth'

    class_size = 6
    NUM_CLASSES = 6
    MAX_SEG_PER_CLASS = 9999
    NUM_WORKERS = 5

    BATCH_SIZE = 102
    GRADIENT_ACCUMULATION = 5
    MAX_EPOCH = 120
    LR = 1e-2
    LR_MIN = 1e-5
    LABEL_SMOOTHING = 0.01
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.1
    DOWNSTREAM_DROPOUT = 0.15
    EMBEDDING_SIZE = 64
    H_FEATS = 128

    DIR_PATH_DICT = {0: r'./dataset/ISCX-NonVPN-2016/process_file/Chat',
                     1: r'./dataset/ISCX-NonVPN-2016/process_file/Email',
                     2: r'./dataset/ISCX-NonVPN-2016/process_file/File',
                     3: r'./dataset/ISCX-NonVPN-2016/process_file/Streaming',
                     4: r'./dataset/ISCX-NonVPN-2016/process_file/Video',
                     5: r'./dataset/ISCX-NonVPN-2016/process_file/VoIP',
                     }


'''
ISCX-Tor Dataset Configuration
'''
class ISCXTorConfig(Config):
    dataset = 'ISCX-Tor-2017'
    TRAIN_DATA = r'./dataset/ISCX-Tor-2017/train_pyg.npz'
    HEADER_TRAIN_DATA = r'./dataset/ISCX-Tor-2017/header_train_pyg.npz'
    
    VAL_DATA = r'./dataset/ISCX-Tor-2017/val_pyg.npz'
    HEADER_VAL_DATA = r'./dataset/ISCX-Tor-2017/header_val_pyg.npz'
    
    TEST_DATA = r'./dataset/ISCX-Tor-2017/test_pyg.npz'
    HEADER_TEST_DATA = r'./dataset/ISCX-Tor-2017/header_test_pyg.npz'



    TRAIN_GRAPH_DATA = r'./dataset/ISCX-Tor-2017/train_graph_pyg.pt'
    HEADER_TRAIN_GRAPH_DATA = r'./dataset/ISCX-Tor-2017/header_train_graph_pyg.pt'
    VAL_GRAPH_DATA = r'./dataset/ISCX-Tor-2017/val_graph_pyg.pt'
    HEADER_VAL_GRAPH_DATA = r'./dataset/ISCX-Tor-2017/header_val_graph_pyg.pt'
    TEST_GRAPH_DATA = r'./dataset/ISCX-Tor-2017/test_graph_pyg.pt'
    HEADER_TEST_GRAPH_DATA = r'./dataset/ISCX-Tor-2017/header_test_graph_pyg.pt'

    
    MIX_MODEL_CHECKPOINT = r'./checkpoints/mix_model_iscx_tor.pth'

    class_size = 8
    NUM_CLASSES = 8
    MAX_SEG_PER_CLASS = 9999
    NUM_WORKERS = 5

    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION = 1
    MAX_EPOCH = 100
    LR = 1e-2
    LR_MIN = 1e-4
    LABEL_SMOOTHING = 0
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.0
    DOWNSTREAM_DROPOUT = 0.0
    EMBEDDING_SIZE = 64
    H_FEATS = 128

    DIR_PATH_DICT = {0: r'./dataset/ISCX-Tor-2017/process_file/Audio-Streaming',
                     1: r'./dataset/ISCX-Tor-2017/process_file/Browsing',
                     2: r'./dataset/ISCX-Tor-2017/process_file/Chat',
                     3: r'./dataset/ISCX-Tor-2017/process_file/File',
                     4: r'./dataset/ISCX-Tor-2017/process_file/Mail',
                     5: r'./dataset/ISCX-Tor-2017/process_file/P2P',
                     6: r'./dataset/ISCX-Tor-2017/process_file/Video-Streaming',
                     7: r'./dataset/ISCX-Tor-2017/process_file/VoIP'
                     }


'''
ISCX-NonTor Dataset Configuration
'''
class ISCXNonTorConfig(Config):
    dataset = 'ISCX-NonTor-2017'
    TRAIN_DATA = r'./dataset/ISCX-NonTor-2017/train_pyg.npz'
    HEADER_TRAIN_DATA = r'./dataset/ISCX-NonTor-2017/header_train_pyg.npz'
    
    VAL_DATA = r'./dataset/ISCX-NonTor-2017/val_pyg.npz'
    HEADER_VAL_DATA = r'./dataset/ISCX-NonTor-2017/header_val_pyg.npz'
    
    TEST_DATA = r'./dataset/ISCX-NonTor-2017/test_pyg.npz'
    HEADER_TEST_DATA = r'./dataset/ISCX-NonTor-2017/header_test_pyg.npz'



    TRAIN_GRAPH_DATA = r'./dataset/ISCX-NonTor-2017/train_graph_pyg.pt'
    HEADER_TRAIN_GRAPH_DATA = r'./dataset/ISCX-NonTor-2017/header_train_graph_pyg.pt'
    VAL_GRAPH_DATA = r'./dataset/ISCX-NonTor-2017/val_graph_pyg.pt'
    HEADER_VAL_GRAPH_DATA = r'./dataset/ISCX-NonTor-2017/header_val_graph_pyg.pt'
    TEST_GRAPH_DATA = r'./dataset/ISCX-NonTor-2017/test_graph_pyg.pt'
    HEADER_TEST_GRAPH_DATA = r'./dataset/ISCX-NonTor-2017/header_test_graph_pyg.pt'


    MIX_MODEL_CHECKPOINT = r'./checkpoints/mix_model_iscx_nontor.pth'

    class_size = 8
    NUM_CLASSES = 8
    MAX_SEG_PER_CLASS = 9999
    NUM_WORKERS = 5

    BATCH_SIZE = 102
    GRADIENT_ACCUMULATION = 5
    MAX_EPOCH = 120
    LR = 1e-2
    LR_MIN = 1e-4
    LABEL_SMOOTHING = 0
    WEIGHT_DECAY = 0
    WARM_UP = 0.1
    SEED = 32
    DROPOUT = 0.05
    DOWNSTREAM_DROPOUT = 0.1
    EMBEDDING_SIZE = 64
    H_FEATS = 128

    DIR_PATH_DICT = {0: r'./dataset/ISCX-NonTor-2017/process_file/Audio',
                     1: r'./dataset/ISCX-NonTor-2017/process_file/Browsing',
                     2: r'./dataset/ISCX-NonTor-2017/process_file/Chat',
                     3: r'./dataset/ISCX-NonTor-2017/process_file/Email',
                     4: r'./dataset/ISCX-NonTor-2017/process_file/FTP',
                     5: r'./dataset/ISCX-NonTor-2017/process_file/P2P',
                     6: r'./dataset/ISCX-NonTor-2017/process_file/Video',
                     7: r'./dataset/ISCX-NonTor-2017/process_file/VoIP',
                     }


if __name__ == '__main__':
    config = Config()
