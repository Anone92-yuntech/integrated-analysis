# Third-party sources

- **OC-SORT**: https://github.com/noahcao/OC_SORT, pinned commit `8462e7e729a93ccd3bd995c0a79a890336cb3a0b`. The original source, MIT license, provenance, and embedded notices are retained under `vendor/ocsort/`.
- **DAN**: https://github.com/yaoing/DAN, source revision `fea9a093bc7c44ee3f08dd00310516b6e5a92c09`. `networks/dan.py` preserves the model architecture, with the torchvision `weights=` compatibility adjustment. The original MIT license is retained as `networks/LICENSE.DAN`.
- **YOLOX**: https://github.com/Megvii-BaseDetection/YOLOX, official model release `0.1.1rc0`. Detector provenance and SHA256 are recorded in `person_detector.py`. Upstream is licensed under Apache-2.0.
- **MediaPipe**: https://github.com/google-ai-edge/mediapipe. The project uses the installed MediaPipe package, its bundled Holistic models, and the Pose Landmarker Lite model. Upstream is licensed under Apache-2.0.
- **Ultralytics**: https://github.com/ultralytics/ultralytics. This project uses the installed package for toy inference. Consult that package's AGPL-3.0 and enterprise licensing terms; the MIT notices above do not relicense this dependency.
- **Local research components**: the integrated tracking, motion, posture, and toy rules derive from the `attention` and `medispipe_final` projects; the binary DAN state dict derives from the `DAN` transfer-learning experiment. Relative source paths and file hashes are retained in `SOURCE_PROVENANCE.json`.

Dependencies and model assets retain their respective upstream terms. This notice does not assert that all components share a single license. No research videos or shared dataset images are redistributed with this repository.
