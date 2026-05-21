# SplitCap Guidance

The ignored local dataset folders may contain dataset-specific `splitcap.sh` files. Those scripts are helper wrappers around `SplitCap.exe`; they are not required by the training code, but they explain the first raw-PCAP preprocessing stage.

## What `splitcap.sh` Does

Each local `splitcap.sh` follows the same pattern:

1. Read raw capture files from `./raw`.
2. Match each filename against class-specific substrings.
3. Run SplitCap in session mode:

```bash
mono SplitCap.exe -r "$file" -s session -o "./process_file/<class>"
```

4. Delete generated UDP session files, because the preprocessing uses TCP sessions only.

The output is a class-organized session-pcap tree:

```text
dataset/<ISCX-name>/process_file/<class>/*.pcap
```

These session pcaps are then converted to per-session `.npz` files and aggregated into the final arrays used by training and explanation.

## Tracked Public Wrapper

For the public repo, use the tracked wrapper instead of the ignored dataset-local shell scripts:

```bash
scripts/splitcap_iscx.sh iscx-vpn
scripts/splitcap_iscx.sh iscx-nonvpn
scripts/splitcap_iscx.sh iscx-tor
scripts/splitcap_iscx.sh iscx-nontor
```

The wrapper calls:

```bash
python3 scripts/preprocess_iscx.py --dataset <name> --stage splitcap --overwrite
```

To run the entire ISCX path from raw PCAPs to final arrays:

```bash
python3 scripts/preprocess_iscx.py --dataset iscx-vpn --stage all --overwrite
```

## Required Layout

Before running SplitCap, prepare:

```text
dataset/<ISCX-name>/raw/
dataset/<ISCX-name>/SplitCap.exe
```

Install `mono` so Linux can execute `SplitCap.exe`.

## Filename Mapping

The class mapping is encoded in `scripts/preprocess_iscx.py`. It is equivalent to the intended `splitcap.sh` behavior:

| Dataset | Example filename substrings | Output classes |
|---|---|---|
| `iscx-vpn` | `chat`, `email`, `bittorrent`, `ftps`, `netflix`, `audio` | Chat, Email, P2P, File, Streaming, VoIP |
| `iscx-nonvpn` | `chat`, `email`, `ftps`, `netflix`, `video`, `audio` | Chat, Email, File, Streaming, Video, VoIP |
| `iscx-tor` | `AUDIO`, `BROWSING`, `CHAT`, `FILE`, `MAIL`, `P2P`, `VIDEO`, `VOIP` | Audio-Streaming, Browsing, Chat, File, Mail, P2P, Video-Streaming, VoIP |
| `iscx-nontor` | `Audio`, `Browsing`, `Chat`, `Email`, `FTP`, `p2p`, `Youtube`, `Voice` | Audio, Browsing, Chat, Email, FTP, P2P, Video, VoIP |

Some local `splitcap.sh` files have mappings commented out from earlier partial runs. The tracked Python mapping is the canonical release mapping.
