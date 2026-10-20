# Annotation guide (Stage 4C)

## What you are labelling
Potted plants photographed on a turntable against a white backdrop, with a
white plastic support frame. You label *pixels*, on the photograph itself.

## Classes

| id | name | include | exclude |
|---:|---|---|---|
| 0 | Background | white backdrop, shadows on the backdrop, anything behind the plant | — |
| 1 | Plant | leaves, stems, petioles, veins, flowers, fruit, and any real plant tissue, **including white or pale tissue** | supports, pot, soil |
| 2 | Pot/Soil | the container and the substrate visible inside it | plant parts overhanging the pot rim |
| 3 | Support | the white plastic stakes, clips, ties and the frame holding the plant | plant parts touching them |
| 4 | Uncertain | pixels you cannot decide (motion blur, occlusion, a thin stem against a bright leaf) | — |

## Rules
1. **White plant tissue is Plant, not Support.** Cotton bolls, white flowers and
   pale stems are the single most common mistake; the support is the plastic
   hardware, not everything white.
2. **Thin structures count.** A stem two pixels wide is still a stem. Zoom in.
3. When plant and support overlap, label the pixel according to what is
   *visible* at that pixel.
4. When in doubt, use **Uncertain** rather than guessing. Uncertain pixels are
   excluded from the IoU/Precision/Recall figures and reported separately, so
   marking them costs nothing and hiding doubt costs accuracy.
5. Do not use the pre-annotations as a reference while painting. They are a
   model's opinion and the point of this benchmark is to measure it.

## Workflow
1. Start the review UI: `python -m agri.plant_gs.paa.stage4c.annotation serve --dir <annotation_dir>`
   and open the printed URL.
2. Enter your name, work through the task list, paint, press Save.
3. Export each saved label with
   `python -m agri.plant_gs.paa.stage4c.annotation import --dir <dir> --task <id> --labels <png> --annotator <name>`
   (the UI writes the PNG for you; import is what makes it count as human).
4. Score: `python -m agri.plant_gs.paa.stage4c.annotation evaluate --dir <dir>`
