#!/usr/bin/env python
"""Stage 4C cross-crop assets, semantics, annotation benchmark and governance.

The package is deliberately layered so that each layer can be tested without a
GPU and without the data:

``guard``       the *decision*. Given an asset entry it answers "what has
                actually been verified, and may this asset be used for
                training?". It never inspects a file by itself, so a caller has
                to pass the evidence in, and an unaudited asset cannot become
                eligible by accident.

``assets``      the *assembly*. Reads the workspaces and the registry, builds
                the unified entry list, and audits identity / hashes / splits.

``annotation``  the *human benchmark*. Exports pending annotation tasks,
                imports a human confirmation, and scores it. Model output can
                never enter the human-label path.

``input_quality`` the *sampling study*. Compares the official uniform sampler
                with a plant-aware one at an identical 1,024-Gaussian budget.

``camera_summary`` the *camera audit*. Turns the per-sample COLMAP audits into
                one cross-crop camera-model table.
"""

from __future__ import annotations

__all__ = ["guard", "assets", "annotation", "input_quality", "camera_summary"]
