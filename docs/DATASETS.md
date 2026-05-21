# Dataset Guide

Traffic-Explainer currently supports six processed datasets:

```text
iscx-vpn
iscx-nonvpn
iscx-tor
iscx-nontor
ios
android
```

The first four are ISCX application-classification datasets. The last two are cross-platform traffic-localization datasets.

## Common Representation

Each sample has two byte sequences:

- `pkt` or body payload sequence
- `head` or packet-header sequence

For ISCX datasets, tensors are stored as `.npz` arrays:

```text
dataset/<dataset_name>/<split>_pyg.npz
dataset/<dataset_name>/header_<split>_pyg.npz
```

The ISCX body shape is `(num_flows, 50, 150)`, meaning up to 50 packets per flow and 150 bytes per packet. The ISCX header shape is `(num_flows, 50, 40)`. Byte values are `0-255`; padding is `256`.

For IOS/Android localization datasets, tensors are stored as `.pkl` dictionaries:

```text
dataset_localization/<dataset_name>/<split>.pkl
```

Each pickle has keys:

```text
pkt
head
label
```

The localization body and header shapes are `(num_samples, 64)`. Values are `0-65535`; padding is `65536`.

## ISCX-VPN-2016

Alias: `iscx-vpn`

Task: classify VPN traffic by application category.

Label mapping:

| ID | Class |
|---:|---|
| 0 | Chat |
| 1 | Email |
| 2 | File |
| 3 | P2P |
| 4 | Streaming |
| 5 | VoIP |

Splits:

| Split | Body Shape | Header Shape | Label Counts |
|---|---:|---:|---|
| train | `(1231, 50, 150)` | `(1231, 50, 40)` | `0:146, 1:110, 2:232, 3:186, 4:220, 5:337` |
| val | `(154, 50, 150)` | `(154, 50, 40)` | `0:18, 1:14, 2:29, 3:23, 4:28, 5:42` |
| test | `(157, 50, 150)` | `(157, 50, 40)` | `0:19, 1:14, 2:29, 3:24, 4:28, 5:43` |

## ISCX-NonVPN-2016

Alias: `iscx-nonvpn`

Task: classify non-VPN traffic by application category.

Label mapping:

| ID | Class |
|---:|---|
| 0 | Chat |
| 1 | Email |
| 2 | File |
| 3 | Streaming |
| 4 | Video |
| 5 | VoIP |

Splits:

| Split | Body Shape | Header Shape | Label Counts |
|---|---:|---:|---|
| train | `(3140, 50, 150)` | `(3140, 50, 40)` | `0:333, 1:188, 2:932, 3:543, 4:421, 5:723` |
| val | `(392, 50, 150)` | `(392, 50, 40)` | `0:42, 1:23, 2:116, 3:68, 4:53, 5:90` |
| test | `(395, 50, 150)` | `(395, 50, 40)` | `0:42, 1:24, 2:117, 3:68, 4:53, 5:91` |

## ISCX-Tor-2017

Alias: `iscx-tor`

Task: classify Tor traffic by application category.

Label mapping:

| ID | Class |
|---:|---|
| 0 | Audio-Streaming |
| 1 | Browsing |
| 2 | Chat |
| 3 | File |
| 4 | Mail |
| 5 | P2P |
| 6 | Video-Streaming |
| 7 | VoIP |

Splits:

| Split | Body Shape | Header Shape | Label Counts |
|---|---:|---:|---|
| train | `(1354, 50, 150)` | `(1354, 50, 40)` | `0:290, 1:331, 2:69, 3:72, 4:42, 5:126, 6:104, 7:320` |
| val | `(169, 50, 150)` | `(169, 50, 40)` | `0:36, 1:41, 2:9, 3:9, 4:5, 5:16, 6:13, 7:40` |
| test | `(174, 50, 150)` | `(174, 50, 40)` | `0:37, 1:42, 2:9, 3:10, 4:6, 5:16, 6:13, 7:41` |

## ISCX-NonTor-2017

Alias: `iscx-nontor`

Task: classify non-Tor traffic by application category.

Label mapping:

| ID | Class |
|---:|---|
| 0 | Audio |
| 1 | Browsing |
| 2 | Chat |
| 3 | Email |
| 4 | FTP |
| 5 | P2P |
| 6 | Video |
| 7 | VoIP |

Splits:

| Split | Body Shape | Header Shape | Label Counts |
|---|---:|---:|---|
| train | `(19179, 50, 150)` | `(19179, 50, 40)` | `0:799, 1:7999, 2:147, 3:116, 4:1258, 5:7999, 6:688, 7:173` |
| val | `(2398, 50, 150)` | `(2398, 50, 40)` | `0:100, 1:1000, 2:18, 3:15, 4:157, 5:1000, 6:86, 7:22` |
| test | `(2400, 50, 150)` | `(2400, 50, 40)` | `0:100, 1:1000, 2:19, 3:15, 4:158, 5:1000, 6:86, 7:22` |

## IOS_Cross_Plat

Alias: `ios`

Task: classify packet samples by country/localization label.

Label mapping:

| ID | Class |
|---:|---|
| 0 | india |
| 1 | china |
| 2 | us |

Splits:

| Split | Body Shape | Header Shape | Label Counts |
|---|---:|---:|---|
| train | `(776156, 64)` | `(776156, 64)` | `0:299572, 1:214536, 2:262048` |
| val | `(97803, 64)` | `(97803, 64)` | `0:36900, 1:28276, 2:32627` |
| test | `(97803, 64)` | `(97803, 64)` | `0:36762, 1:28517, 2:32524` |

The raw CSV contains `train`, `dev`, and `test` rows. The processing script keeps `train` as training data and shuffles all non-train rows into validation and test halves.

## Android_Cross_Plat

Alias: `android`

Task: classify packet samples by country/localization label.

Label mapping:

| ID | Class |
|---:|---|
| 0 | india |
| 1 | china |
| 2 | us |

Splits:

| Split | Body Shape | Header Shape | Label Counts |
|---|---:|---:|---|
| train | `(1086909, 64)` | `(1086909, 64)` | `0:94809, 1:893483, 2:98617` |
| val | `(135691, 64)` | `(135691, 64)` | `0:12626, 1:110233, 2:12832` |
| test | `(135692, 64)` | `(135692, 64)` | `0:12409, 1:110506, 2:12777` |

The raw CSV contains `train`, `dev`, and `test` rows. The processing script keeps `train` as training data and shuffles all non-train rows into validation and test halves.

## Local But Not Currently Supported

The local `dataset_localization/USTC-TFC2016/` folder exists, but `config.py` does not currently define a public dataset alias or config class for it. It should be treated as an experimental local artifact unless support is added explicitly.
