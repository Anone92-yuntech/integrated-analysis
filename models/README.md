# Model assets

The four inference models are distributed as assets of the
[v1.0.0 Release](https://github.com/Anone92-yuntech/integrated-analysis/releases/tag/v1.0.0).
Model binaries are excluded from Git. All four files belong in this `models/`
directory; no sibling project or dataset directory is required.

From the project root, download and verify them with Python's standard library:

```powershell
python fetch_models.py
python fetch_models.py --verify-only
```

If using the virtual environment described in the main README:

```powershell
.\.venv\Scripts\python.exe fetch_models.py
.\.venv\Scripts\python.exe fetch_models.py --verify-only
```

| Release asset | Local path | Bytes | SHA256 |
| --- | --- | ---: | --- |
| best.pt | models/best.pt | 40516645 | f5fce7480e8fb4aebd800eeaae24299979cd7aa6df49c8085b0a67ee6ecf906f |
| best_transfer_model.pth | models/best_transfer_model.pth | 79069865 | fccef63f6464c678852224f7cbf24c9c6fb9b9e682f0fd15f774ec7df4b245b3 |
| yolox_nano.onnx | models/yolox_nano.onnx | 3659407 | c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d |
| pose_landmarker_lite.task | models/pose_landmarker_lite.task | 5777746 | 59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a |

`best.pt` is the four-class toy detector (`toy`, `basic`, `toy3D`, `basic3D`).
`best_transfer_model.pth` is the transferred DAN classifier for neutral/positive
facial expression (`0=neutral`, `1=positive`); inference code `2` means missing
evidence and is not a third trained class.

YOLOX-Nano is the official COCO model from the
[YOLOX 0.1.1rc0 Release](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0).
Pose Landmarker Lite is the
[official MediaPipe float16 model](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task).
The pinned files are unchanged copies of their recorded source assets. Source
hashes and code provenance are documented in `../SOURCE_PROVENANCE.json`.
Third-party source license notices remain in `../vendor/ocsort/` and
`../networks/LICENSE.DAN`.

The downloader uses public HTTPS URLs without authentication, checks both size
and SHA256, and publishes a complete model atomically only after verification.
Correct existing files are reused without network access. A wrong existing file,
directory or symbolic link is left unchanged and reported as an error; the tool
never silently repairs or overwrites it. Interrupted or invalid new downloads
remove only their own temporary `.part` file. Atomic publication requires a
filesystem supporting hard links, such as the usual Windows NTFS installation.

`--verify-only` is read-only and returns a nonzero exit status if a model is
missing or fails verification. Tests use tiny in-memory responses; running the
test suite does not download these assets.

Input videos and analysis output are not distributed. Supply your own video
using `run_integrated.cmd --input 'D:\videos\example.mp4'` after installation.
