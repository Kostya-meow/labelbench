# Robustness and evidence fusion experiment

Build a standalone, resumable SQLite-backed worker. Transform one image at a time; never retain augmented images. Process all 1000 source photos, evaluate only the 977 pages with complete polygon GT. Keep the existing inference API compatible.

- [x] Literature-backed protocol: frozen detectors, corruption/TTA evidence, family-aware consensus and mask voting.
- [x] Deterministic transforms with inverse geometry, evidence selection, multi-metric evaluation.
- [x] SQLite checkpoints per model/view and per page, worker heartbeat/cancel/resume, fixed configuration and environment.
- [x] Live dashboard, ETA, comparison plots, page overlays and downloadable analysis.
- [x] Unit/integration/browser checks, real small pilot including restart/cancel. 82 Python tests, JS overlay test, responsive browser check. Final pilot 760f1da02735: 108/108 tasks, 3 pages, 0 errors; 2 GT pages and 1 intentionally excluded from scoring.
- [ ] Freeze protocol, commit, launch full background experiment, verify live progress.

No parameter selection on test GT; agreement is not truth. Document/page dependence limits page-bootstrap CIs. The experiment uses pretrained networks without target-data fine-tuning, not untrained networks.
