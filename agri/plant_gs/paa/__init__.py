"""PAA (Plant Architecture Analysis) ingestion for the AgriGaussian-JEPA pipeline.

The package turns a raw HuggingFace `PAA-Plant-Dataset` sample into exactly the
workspace layout that the already-validated `agri/plant_gs/foreground` pipeline
consumes, so the 3DGS trainer, exporter and Gaussian-JEPA checks are reused
rather than reimplemented.

The one thing PAA does not hand over ready-made is the *mask registration*: PAA
ships `Camera/images` (the frames COLMAP was solved on) and `RGB_seg/masks` in
different orientations and at different resolutions. `alignment` recovers that
mapping from the data itself and reports an IoU, so the correspondence is a
verified measurement rather than an assumption.
"""
