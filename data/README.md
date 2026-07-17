# Dataset Layout

Place training images using this structure:

```text
data/
├── train/
│   ├── real/
│   └── fake_face_swap/
└── val/
    ├── real/
    └── fake_face_swap/
```

Quick demo dataset:

```bash
python scripts/generate_demo_dataset.py
python training/train.py
```

This project targets **face-swap spatial artifacts**, not all modern generative video models.
