"""Batch registry for PAA plant reconstruction and Gaussian-JEPA acceptance.

The schema exists to prevent one specific mistake: **treating one acquisition
of a plant as a different plant.**

PAA's dated samples follow ``<YYYYMMDD>-<plant_id>``. Broccoli has 8 samples
over four dates at two plant IDs (``5`` and ``10``), so ``20260327-10`` and
``20260402-10`` are the *same physical plant* photographed a week apart. If a
future train/val/test split grouped by sample directory, it would put the same
plant on both sides of the split and report a generalisation number that is
really memorisation of that plant's silhouette. The registry therefore keeps
two separate keys and exposes :func:`group_key` so splitting code has one
obvious correct thing to call.

Samples that are not dated (Cotton/Rice/Sunflower use bare ``1``..``5``) have
no acquisition date; their plant id is the sample name and they carry exactly
one acquisition. :func:`parse_sample_id` handles both shapes and refuses to
guess when the format is neither.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = "paa-stage4b-1"

# Fields the brief asks for, kept in one place so the writer, the validator and
# the tests cannot drift apart.
REQUIRED_FIELDS = (
    "dataset_name", "crop", "plant_id", "acquisition_date", "acquisition_id",
    "source_url", "source_revision", "license", "colmap_status",
    "gaussian_ply_path", "gaussian_count", "reconstruction_quality",
    "jepa_compatibility", "acceptance_status",
)

ACCEPTANCE_STATES = ("accepted", "rejected", "pending")
COLMAP_STATES = ("pass", "fail", "not_run")
JEPA_STATES = ("pass", "fail", "not_run")
#: What an acceptance decision saw of the checkpoint -> PLY chain.
PROVENANCE_STATES = ("verified", "failed", "absent", "legacy")

_DATED = re.compile(r"^(?P<date>\d{8})-(?P<plant>.+)$")


def parse_sample_id(sample: str) -> dict:
    """Split a PAA sample directory name into plant identity and acquisition.

    ``"20260327-10"`` -> date 2026-03-27, plant ``10``, dated acquisition.
    ``"3"``           -> no date, plant ``3``, undated acquisition.
    """
    match = _DATED.match(sample)
    if match:
        raw = match.group("date")
        return {
            "plant_id": match.group("plant"),
            "acquisition_id": sample,
            "acquisition_date": f"{raw[:4]}-{raw[4:6]}-{raw[6:]}",
            "dated": True,
        }
    if sample and not any(ch.isdigit() for ch in sample.split("-")[0]):
        raise ValueError(f"cannot parse PAA sample id: {sample!r}")
    return {
        "plant_id": sample,
        "acquisition_id": sample,
        "acquisition_date": None,
        "dated": False,
    }


def group_key(crop: str, plant_id: str) -> str:
    """The key a train/val/test split MUST group by.

    Splitting by acquisition leaks plants across dates. Splitting by this key
    keeps every acquisition of one plant on the same side.
    """
    return f"{crop}/{plant_id}"


@dataclass
class PlantRecord:
    """One reconstructed PAA sample, with everything needed to audit it."""

    dataset_name: str = "PAA-Plant-Dataset"
    crop: str = ""
    plant_id: str = ""
    acquisition_date: str | None = None
    acquisition_id: str = ""
    sample_id: str = ""
    source_url: str = "https://huggingface.co/datasets/JackTrum938/PAA-Plant-Dataset"
    source_revision: str | None = None
    license: str = "apache-2.0"
    colmap_status: str = "not_run"
    gaussian_ply_path: str | None = None
    gaussian_count: int | None = None
    reconstruction_quality: dict = field(default_factory=dict)
    jepa_compatibility: dict = field(default_factory=dict)
    acceptance_status: str = "pending"
    # provenance / audit extras
    schema_version: str = SCHEMA_VERSION
    sample_dir: str | None = None
    source: dict = field(default_factory=dict)
    registered_images: int | None = None
    frames_used: int | None = None
    sparse_points: int | None = None
    reprojection_error_mean_px: float | None = None
    mask_hit_rate_median: float | None = None
    training_steps: int | None = None
    created_utc: str | None = None
    # The checkpoint -> PLY chain: hashes, the step inside the checkpoint, and
    # the outcome of the independent re-check. An `accepted` row requires this
    # to have verified, so a number cannot be accepted on a report's word alone.
    provenance: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def split_group(self) -> str:
        return group_key(self.crop, self.plant_id)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["split_group"] = self.split_group
        return payload

    def validate(self) -> list:
        """Return a list of problems; empty means the record is well formed."""
        problems = []
        missing = [f for f in REQUIRED_FIELDS if f not in self.to_dict()]
        if missing:
            problems.append(f"missing required fields: {missing}")
        if not self.crop:
            problems.append("crop is empty")
        if not self.plant_id:
            problems.append("plant_id is empty")
        if self.colmap_status not in COLMAP_STATES:
            problems.append(f"colmap_status {self.colmap_status!r} not in {COLMAP_STATES}")
        if self.acceptance_status not in ACCEPTANCE_STATES:
            problems.append(f"acceptance_status {self.acceptance_status!r} not in {ACCEPTANCE_STATES}")
        if self.jepa_compatibility.get("status") not in tuple(JEPA_STATES) + (None,):
            problems.append("jepa_compatibility.status invalid")
        # An accepted record must carry the evidence that justifies it.
        if self.acceptance_status == "accepted":
            if self.colmap_status != "pass":
                problems.append("accepted record has colmap_status != pass")
            if not self.gaussian_ply_path:
                problems.append("accepted record has no gaussian_ply_path")
            if not self.gaussian_count:
                problems.append("accepted record has no gaussian_count")
            if self.jepa_compatibility.get("status") != "pass":
                problems.append("accepted record has jepa_compatibility.status != pass")
            # Stage 4B-Fix: the metric must be attributable to a checkpoint.
            if self.provenance.get("state") != "verified":
                problems.append(
                    "accepted record has no verified checkpoint->PLY provenance "
                    "(state=%r); a metric whose source checkpoint cannot be "
                    "identified is not acceptable" % self.provenance.get("state"))
        elif self.provenance.get("state") not in PROVENANCE_STATES + (None,):
            problems.append(f"provenance.state {self.provenance.get('state')!r} not in "
                            f"{PROVENANCE_STATES}")
        return problems


def record_from_workspace(workspace: Path, sample: str, crop: str,
                          revision: str | None = None,
                          source: dict | None = None) -> PlantRecord:
    """Assemble a record from the artefacts a pilot run leaves on disk.

    Reads whatever is present and records what is missing rather than filling
    it in. A record built from a half-finished run is `pending`, never
    `accepted`.

    ``source`` carries the upstream provenance (Hub revision, file counts,
    licence) so the registry row is self-describing instead of pointing at a
    JSON that may live outside the repository.
    """
    workspace = Path(workspace)
    parsed = parse_sample_id(sample)
    record = PlantRecord(
        crop=crop, plant_id=parsed["plant_id"],
        acquisition_id=parsed["acquisition_id"],
        acquisition_date=parsed["acquisition_date"], sample_id=sample,
        source_revision=revision, sample_dir=str(workspace.resolve()),
        source=dict(source or {}),
    )

    preparation = workspace / "preparation.json"
    if preparation.is_file():
        meta = json.loads(preparation.read_text())
        record.registered_images = meta.get("registered_images")
        record.frames_used = meta.get("frames_used")
        record.sparse_points = meta.get("sparse_points_plant")
        record.reprojection_error_mean_px = meta.get("reprojection_error_mean_px")
        record.mask_hit_rate_median = meta.get("mask_hit_rate_median")
        record.training_steps = None
    audit = workspace / "colmap_audit.json"
    if audit.is_file():
        record.colmap_status = "pass" if json.loads(audit.read_text()).get(
            "status") == "PASS" else "fail"

    report = workspace / "final" / "report.json"
    if report.is_file():
        payload = json.loads(report.read_text())
        record.gaussian_ply_path = payload.get("point_cloud")
        record.gaussian_count = payload.get("retained_gaussians")
        record.training_steps = payload.get("step")
        clean = payload.get("clean_validation", {})
        record.reconstruction_quality = {
            "foreground_psnr": clean.get("foreground_psnr"),
            "silhouette_iou": clean.get("silhouette_iou"),
            "background_alpha": clean.get("background_alpha"),
            "raw_gaussians": payload.get("raw_gaussians"),
            "retained_gaussians": payload.get("retained_gaussians"),
            "held_out_scope": "single-scene held-out views; not plant generalisation",
        }

    # The compatibility report is written next to the PLY by
    # verify_gsjepa_compatible.py, which the pilot script points at the export
    # directory. Look in both the export dir and the workspace root so a run
    # that wrote it elsewhere is still picked up rather than silently scored
    # as "not run".
    compat = None
    for candidate in (workspace / "final" / "gsjepa_compatibility.json",
                      workspace / "gsjepa_compatibility.json",
                      workspace / "final" / "gsjepa_compatibility.json"):
        if candidate.is_file():
            compat = candidate
            break
    if compat is not None:
        payload = json.loads(compat.read_text())
        record.jepa_compatibility = {
            "status": "pass" if payload.get("gsjepa_compatible") else "fail",
            "attributes_present": payload.get("required_fields"),
            "detail": str(compat),
        }

    # Acceptance is earned, not assumed. Record every missing piece of
    # evidence by name, so a `pending` row says WHY it is pending instead of
    # forcing whoever reads it next to re-derive the reason from blanks.
    #
    # The checkpoint -> PLY chain is read from the verification the exporter's
    # independent checker wrote. It is not re-derived here: re-running the
    # check during registration would make the row depend on whether the
    # checkpoint is still present, rather than on what the run actually proved.
    verification_path = workspace / "final" / "provenance_verification.json"
    verification = None
    if verification_path.is_file():
        try:
            verification = json.loads(verification_path.read_text())
        except Exception as exc:
            verification = {"unreadable": str(exc)}

    if verification is None:
        record.provenance = {"state": "absent", "record": None,
                             "detail": "no final/provenance_verification.json"}
    elif "unreadable" in verification:
        record.provenance = {"state": "failed", "record": str(verification_path),
                             "detail": verification["unreadable"]}
    else:
        record.provenance = {
            "state": "verified" if verification.get("status") == "pass" else "failed",
            "record": str(verification_path.resolve()),
            "checkpoint_sha256": (verification.get("details") or {}).get("checkpoint_sha256"),
            "ply_sha256": (verification.get("details") or {}).get("ply_sha256"),
            "checkpoint_step": (verification.get("details") or {}).get("checkpoint_step"),
            "gaussians_after_filter": (verification.get("details") or {}).get(
                "gaussians_after_filter"),
            "problems": verification.get("problems") or [],
        }

    blocking = []
    if record.colmap_status != "pass":
        blocking.append(f"colmap_status={record.colmap_status}")
    if not record.gaussian_ply_path:
        blocking.append("no gaussian_ply_path")
    if not record.gaussian_count:
        blocking.append("no gaussian_count")
    if record.jepa_compatibility.get("status") != "pass":
        blocking.append(f"jepa_compatibility={record.jepa_compatibility.get('status')}")
    if record.provenance.get("state") != "verified":
        blocking.append(f"provenance={record.provenance.get('state')}")

    if not blocking:
        record.acceptance_status = "accepted"
    elif record.colmap_status == "fail":
        record.acceptance_status = "rejected"
    else:
        record.acceptance_status = "pending"
    record.notes.extend(blocking)
    record.notes.extend(record.validate())
    return record


def record_key(crop: str, acquisition_id: str) -> str:
    """Identity of one *acquisition* -- what a re-registration must collapse to.

    Plant identity is a separate axis (``split_group``); one plant has many
    acquisitions. A registry keyed only by plant would lose a date; keyed only
    by sample directory it would duplicate. ``crop/acquisition`` is the
    acquisition, which is what a rebuild of the same run must overwrite.
    """
    return f"{crop}/{acquisition_id}"


class Registry:
    """JSONL registry of plant records, one row per acquisition.

    Re-registering an acquisition that is already present **replaces** its row
    rather than appending a second one. The previous row is not discarded: its
    revision and acceptance are kept under ``history`` so a later reader can
    see that the run was rebuilt and from which source revision each build came.
    """

    def __init__(self, path: Path):
        self.path = Path(path)

    def read(self) -> list:
        if not self.path.is_file():
            return []
        rows = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    def latest(self) -> dict:
        """Key -> most recent row, collapsing duplicate acquisitions."""
        out = {}
        for row in self.read():
            key = record_key(row.get("crop", ""), row.get("acquisition_id", ""))
            out[key] = row
        return out

    def append(self, record: PlantRecord) -> None:
        """Deprecated shim: writes a new row without deduplication."""
        problems = record.validate()
        if problems:
            raise ValueError(f"refusing to register an invalid record: {problems}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as stream:
            stream.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def upsert(self, record: PlantRecord) -> dict:
        """Register ``record``, replacing any earlier row for the same acquisition.

        Rewrites the file in place. If the acquisition is already registered the
        earlier row is archived into the new row's ``history`` (bounded to the
        last few builds) and the new row takes its place, so the file stays one
        line per acquisition. Returns a small summary of what happened.
        """
        problems = record.validate()
        if problems:
            raise ValueError(f"refusing to register an invalid record: {problems}")

        payload = record.to_dict()
        key = record_key(record.crop, record.acquisition_id)
        rows = self.read()
        replaced = 0
        kept = []
        for row in rows:
            if record_key(row.get("crop", ""), row.get("acquisition_id", "")) == key:
                replaced += 1
                history = list(row.get("history", []))
                history.append({
                    "schema_version": row.get("schema_version"),
                    "source_revision": row.get("source_revision"),
                    "gaussian_count": row.get("gaussian_count"),
                    "acceptance_status": row.get("acceptance_status"),
                    "created_utc": row.get("created_utc"),
                })
                payload["history"] = history[-5:]
                continue
            kept.append(row)

        kept.append(payload)
        kept.sort(key=lambda r: (r.get("crop", ""), r.get("acquisition_id", "")))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w") as stream:
            for row in kept:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(self.path)

        # A row for the same acquisition that predates deduplication support
        # would have shown up as `replaced`; anything else is a first write.
        return {"key": key, "replaced": replaced, "rows": len(kept),
                "status": "updated" if replaced else "inserted"}

    def groups(self) -> dict:
        """plant identity -> acquisitions, the shape a splitter needs."""
        out: dict[str, list] = {}
        for row in self.read():
            out.setdefault(row["split_group"], []).append(row["acquisition_id"])
        return {k: sorted(set(v)) for k, v in sorted(out.items())}

    def check_split(self, train_groups, test_groups) -> list:
        """Return every plant identity that appears on both sides."""
        return sorted(set(train_groups) & set(test_groups))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, help="a completed pilot workspace")
    parser.add_argument("--sample", help="PAA sample id, e.g. 20260327-10")
    parser.add_argument("--crop", help="crop name, e.g. Broccoli")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--provenance", type=Path, default=None,
                        help="the sample's provenance.json (Hub revision, file counts)")
    parser.add_argument("--registry", type=Path, default=None,
                        help="JSONL registry to append to")
    parser.add_argument("--parse", help="just parse a sample id and print the split")
    parser.add_argument("--dedupe", action="store_true",
                        help="collapse duplicate acquisitions in an existing registry "
                             "(keeps the most recent row per crop/acquisition)")
    args = parser.parse_args()

    if args.parse:
        parsed = parse_sample_id(args.parse)
        crop = args.crop or "<crop>"
        print(json.dumps({**parsed, "split_group": group_key(crop, parsed["plant_id"])},
                         indent=2))
        return
    if args.dedupe:
        if not args.registry:
            parser.error("--dedupe needs --registry")
        registry = Registry(args.registry)
        rows = registry.read()
        latest = registry.latest()
        registry.path.write_text("".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in sorted(latest.values(),
                              key=lambda r: (r.get("crop", ""), r.get("acquisition_id", "")))))
        print(json.dumps({"rows_before": len(rows), "rows_after": len(latest)},
                         indent=2))
        return
    if not (args.workspace and args.sample and args.crop):
        parser.error("--workspace, --sample and --crop are required unless --parse is used")

    source = {}
    revision = args.revision
    if args.provenance and Path(args.provenance).is_file():
        payload = json.loads(Path(args.provenance).read_text())
        source = {
            "provenance_path": str(Path(args.provenance).resolve()),
            "hf_endpoint": payload.get("hf_endpoint_used"),
            "hub_sha": (payload.get("hub") or {}).get("sha"),
            "hub_last_modified": (payload.get("hub") or {}).get("last_modified"),
            "file_counts": payload.get("file_counts"),
            "downloaded_at": payload.get("downloaded_at"),
        }
        revision = revision or (payload.get("hub") or {}).get("sha")

    record = record_from_workspace(args.workspace, args.sample, args.crop,
                                   revision, source=source)
    print(json.dumps(record.to_dict(), indent=2, ensure_ascii=False))
    if args.registry:
        outcome = Registry(args.registry).upsert(record)
        print(f"registry {outcome['status']} {outcome['key']} "
              f"({outcome['rows']} rows): {args.registry}")


if __name__ == "__main__":
    main()