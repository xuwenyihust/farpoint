# Dataset quality reports

Farpoint dataset quality reports are immutable, precomputed views over a published
LeRobot dataset revision. The analyzer never reads a mutable Hub branch when producing
a release report, and generated reports and visual assets remain outside Git.

The `farpoint.dataset-quality-report.v1` contract contains five public sections:

1. Overview
2. Integrity
3. Variation Coverage
4. Episode & Action Quality
5. Visual Quality

Generate a report from a local copy of the exact published package:

```bash
python scripts/generate_dataset_quality_report.py \
  --dataset-root /path/to/public-package \
  --output /path/to/quality-report \
  --dataset-repo owner/dataset \
  --dataset-tag v0.0.3 \
  --resolved-dataset-commit 0123456789abcdef0123456789abcdef01234567 \
  --generator-commit 89abcdef0123456789abcdef0123456789abcdef
```

The report records hashes for every analyzed data, metadata, and video file. Integrity
checks include episode and frame counts, successful outcomes, finite state/action values,
feature dimensions, monotonic timestamps, terminal markers, full video decoding,
resolution, and logical splits. Optional published-tag and Dataset Viewer evidence can be
attached by passing their validation JSON files.

Stage the static Hugging Face Space from a PASS report:

```bash
python scripts/stage_dataset_quality_space.py \
  --template spaces/farpoint-so101-dataset \
  --report /path/to/quality-report \
  --output /path/to/staged-space
```

The staging audit verifies the report hash, integrity status, version index, and every
referenced visual asset. The Space does not download or analyze the source dataset at
runtime. Add future dataset versions by regenerating reports from their immutable tags and
extending the version index; do not rewrite an existing report version in place.
