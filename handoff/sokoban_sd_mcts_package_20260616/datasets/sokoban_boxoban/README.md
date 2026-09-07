# Sokoban / Boxoban Dataset

Downloaded dataset: Google DeepMind `boxoban-levels`.

Source repository: https://github.com/google-deepmind/boxoban-levels.git

Local raw data path:

- `datasets/sokoban_boxoban/boxoban-levels/`

Download date: 2026-06-10

Git commit:

```text
78011d8c86ad423bfb2e1005012393b4c3c24409
```

License: Apache License 2.0, as provided by the upstream repository.

Dataset summary verified locally:

- Size on disk: approximately 204 MB
- Text files: 1506
- Total level records: 1,504,332
- Format: 10x10 ASCII Sokoban-style levels
- Splits:
  - `unfiltered/train`: 900 files, 900,000 levels
  - `unfiltered/valid`: 100 files, 100,000 levels
  - `unfiltered/test`: 1 file, 1,000 levels
  - `medium/train`: 450 files, 450,000 levels
  - `medium/valid`: 50 files, 50,000 levels
  - `hard`: 4 files, 3,332 levels

Upstream citation:

```bibtex
@misc{boxobanlevels,
author = {Arthur Guez and Mehdi Mirza and Karol Gregor and Rishabh Kabra and Sebastien Racaniere and Theophane Weber and David Raposo and Adam Santoro and Laurent Orseau and Tom Eccles and Greg Wayne and David Silver and Timothy Lillicrap and Victor Valdes},
title = {An investigation of Model-free planning: boxoban levels},
howpublished= {https://github.com/deepmind/boxoban-levels/},
year = "2018"
}
```

Notes:

- The raw dataset is kept as an upstream shallow Git clone for provenance.
- Do not edit files under `boxoban-levels/` directly. Put derived subsets, parsed tensors, or experiment outputs in separate generated-data directories.
