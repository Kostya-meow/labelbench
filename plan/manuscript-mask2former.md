# Manuscript Mask2Former integration

- [x] Inspect exact branch v_0_1_13, commit 1bc3b501d3f3124cc166613b279c463dadc3ef6d. Model is ONNX-native, not a PyTorch export task.
- [x] Create isolated .venv-manuscript with editable local checkout, no PyPI manuscript package.
- [x] Download and hash-verify branch model bundle; verify real CUDA inference.
- [x] Connect independent provider, confidence, API, UI and reproducible setup batch files.
- [x] Run upstream Mask2Former tests, integration/API/browser checks and regression suite.

User-reported Einsum allocation failure: replace ten mask einsums with equivalent MatMul graph, preserve original release files, gate on GPU parity; outputs exactly equal on two archive pages. Exact ROI-based mask IoU speeds up upstream NMS. API test: 94 polygons in 12.79 seconds including worker startup; cache reuse verified.

Previous robustness run is cancelled (174 tasks, 4 pages, no errors); do not restart it implicitly.
