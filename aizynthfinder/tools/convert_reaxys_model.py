"""Convert the licensed Reaxys TorchServe archive to AiZynthFinder assets.

The conversion-only dependencies (PyTorch, ONNX and ONNX Runtime) are imported
inside :func:`convert_reaxys_archive`. They are deliberately not runtime
dependencies of AiZynthFinder, which only needs ONNX Runtime to use the
converted policy.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


EXPECTED_ARCHIVE_SHA256 = (
    "fb05bb216df2c6483bccc738dbc0e5b17a9d6ce3c4f07f4ad8dcda7a0c7ac526"
)
EXPECTED_TEMPLATE_COUNT = 163_723
EXPECTED_FP_SIZE = 2_048
EXPECTED_RADIUS = 2
EXPECTED_HIDDEN_SIZES = [300, 300, 300, 300, 300]
EXPECTED_TOP_INDICES = np.asarray(
    [3657, 41, 5, 219, 35, 14177, 48, 810, 1272, 10552], dtype=np.int64
)
EXPECTED_TOP_SCORES = np.asarray(
    [
        0.1225668415,
        0.1138394102,
        0.0840509385,
        0.0713936761,
        0.0437196456,
        0.0400650315,
        0.0347193219,
        0.0341283493,
        0.0251520444,
        0.0245245919,
    ],
    dtype=np.float32,
)
BENCHMARK_SMILES = "CC(C)(C)OC(=O)N1CCC(OCCO)CC1"
TEMPLATE_COLUMNS = (
    "template_code",
    "retro_template",
    "count",
    "dimer_only",
    "intra_only",
    "necessary_reagent",
    "template_set",
    "_id",
)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fileobj:
        for chunk in iter(lambda: fileobj.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parenthesize_template(smarts: str) -> str:
    """Apply the bookkeeping transformation used by the original handler."""
    if smarts.count(">>") != 1:
        raise ValueError(f"Expected one reaction arrow in template: {smarts}")
    return "(" + smarts.replace(">>", ")>>(") + ")"


def _convert_templates(
    archive: zipfile.ZipFile,
    output_path: Path,
    expected_count: int = EXPECTED_TEMPLATE_COUNT,
) -> int:
    """Convert the JSONL template library to AiZynthFinder's TSV format."""
    with output_path.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_output, mtime=0
        ) as compressed:
            with io.TextIOWrapper(
                compressed, encoding="utf-8", newline=""
            ) as text_output:
                writer = csv.DictWriter(
                    text_output,
                    fieldnames=TEMPLATE_COLUMNS,
                    delimiter="\t",
                    lineterminator="\n",
                )
                writer.writeheader()
                count = 0
                with archive.open("templates.jsonl") as template_stream:
                    for count, raw_line in enumerate(template_stream, start=1):
                        template = json.loads(raw_line)
                        template_index = count - 1
                        if template.get("index") != template_index:
                            raise ValueError(
                                "Template indices must be contiguous and match model "
                                f"outputs: expected {template_index}, got "
                                f"{template.get('index')}"
                            )
                        for field in (
                            "reaction_smarts",
                            "count",
                            "dimer_only",
                            "intra_only",
                            "_id",
                        ):
                            if field not in template:
                                raise ValueError(
                                    f"Template {template_index} is missing {field!r}"
                                )
                        writer.writerow(
                            {
                                "template_code": template_index,
                                "retro_template": _parenthesize_template(
                                    template["reaction_smarts"]
                                ),
                                "count": template["count"],
                                "dimer_only": template["dimer_only"],
                                "intra_only": template["intra_only"],
                                "necessary_reagent": template.get(
                                    "necessary_reagent", ""
                                ),
                                "template_set": template.get("template_set", "reaxys"),
                                "_id": template["_id"],
                            }
                        )

    if count != expected_count:
        raise ValueError(
            f"Expected {expected_count} templates in archive, found {count}"
        )
    return count


def _load_conversion_dependencies() -> Tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import onnx
        import onnxruntime
        import torch
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except ImportError as err:
        raise RuntimeError(
            "Converting Reaxys requires torch, onnx, onnxruntime and rdkit. "
            "Use the documented disposable conversion container."
        ) from err
    return torch, onnx, onnxruntime, Chem, DataStructs, AllChem


def _load_checkpoint(torch: Any, archive: zipfile.ZipFile) -> Dict[str, Any]:
    checkpoint_buffer = io.BytesIO(archive.read("model_latest.pt"))
    try:
        checkpoint = torch.load(
            checkpoint_buffer, map_location=torch.device("cpu"), weights_only=False
        )
    except TypeError:
        checkpoint_buffer.seek(0)
        checkpoint = torch.load(checkpoint_buffer, map_location=torch.device("cpu"))
    if set(checkpoint) != {"args", "state_dict"}:
        raise ValueError(
            "Unexpected checkpoint members: " + ", ".join(sorted(checkpoint))
        )
    return checkpoint


def _build_probability_model(torch: Any, checkpoint: Dict[str, Any]) -> Any:
    args = checkpoint["args"]
    hidden_sizes = args.hidden_sizes
    if isinstance(hidden_sizes, str):
        hidden_sizes = [int(size) for size in hidden_sizes.split(",")]

    expected_settings = {
        "fp_size": EXPECTED_FP_SIZE,
        "radius": EXPECTED_RADIUS,
        "n_templates": EXPECTED_TEMPLATE_COUNT,
        "hidden_sizes": EXPECTED_HIDDEN_SIZES,
        "hidden_activation": "relu",
        "skip_connection": "none",
    }
    actual_settings = {
        "fp_size": args.fp_size,
        "radius": args.radius,
        "n_templates": args.n_templates,
        "hidden_sizes": hidden_sizes,
        "hidden_activation": args.hidden_activation,
        "skip_connection": args.skip_connection,
    }
    if actual_settings != expected_settings:
        raise ValueError(
            "Unexpected Reaxys checkpoint architecture: "
            f"expected {expected_settings}, got {actual_settings}"
        )

    class Dense(torch.nn.Module):
        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(in_features, out_features, bias=True)
            self.hidden_act = torch.nn.ReLU()

        def forward(self, inputs: Any) -> Any:
            return self.hidden_act(self.linear(inputs))

    class TemplateRelevanceModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layer_sizes = [args.fp_size] + hidden_sizes
            self.layers = torch.nn.ModuleList(
                [
                    Dense(in_features, out_features)
                    for in_features, out_features in zip(
                        layer_sizes[:-1], layer_sizes[1:]
                    )
                ]
            )
            self.output_layer = torch.nn.Linear(
                hidden_sizes[-1], args.n_templates, bias=True
            )
            self.dropout = torch.nn.Dropout(args.dropout)

        def forward(self, inputs: Any) -> Any:
            for layer in self.layers:
                inputs = self.dropout(layer(inputs))
            return self.output_layer(inputs)

    class ProbabilityModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = TemplateRelevanceModel()

        def forward(self, inputs: Any) -> Any:
            return torch.softmax(self.model(inputs), dim=1)

    probability_model = ProbabilityModel()
    state_dict = {
        key.replace("module.", ""): value
        for key, value in checkpoint["state_dict"].items()
    }
    probability_model.model.load_state_dict(state_dict)
    probability_model.eval()
    return probability_model


def _make_benchmark_fingerprint(
    Chem: Any, DataStructs: Any, AllChem: Any
) -> np.ndarray:
    molecule = Chem.MolFromSmiles(BENCHMARK_SMILES)
    if molecule is None:
        raise ValueError(f"Could not parse benchmark SMILES: {BENCHMARK_SMILES}")
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        EXPECTED_RADIUS,
        nBits=EXPECTED_FP_SIZE,
        useChirality=True,
    )
    array = np.zeros((EXPECTED_FP_SIZE,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array.reshape(1, EXPECTED_FP_SIZE)


def _export_and_validate_model(
    model: Any,
    output_path: Path,
    dependencies: Tuple[Any, Any, Any, Any, Any, Any],
) -> Sequence[float]:
    torch, onnx, onnxruntime, Chem, DataStructs, AllChem = dependencies
    dummy_input = torch.zeros((1, EXPECTED_FP_SIZE), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["fingerprint"],
        output_names=["probabilities"],
        dynamic_axes={
            "fingerprint": {0: "batch"},
            "probabilities": {0: "batch"},
        },
    )
    onnx.checker.check_model(str(output_path))

    fingerprint = _make_benchmark_fingerprint(Chem, DataStructs, AllChem)
    with torch.no_grad():
        torch_output = model(torch.from_numpy(fingerprint)).cpu().numpy()

    session = onnxruntime.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )
    onnx_output = session.run(
        [session.get_outputs()[0].name],
        {session.get_inputs()[0].name: fingerprint},
    )[0]
    if onnx_output.shape != (1, EXPECTED_TEMPLATE_COUNT):
        raise ValueError(f"Unexpected ONNX output shape: {onnx_output.shape}")
    np.testing.assert_allclose(onnx_output, torch_output, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(onnx_output.sum(axis=1), [1.0], atol=1e-5)

    top_indices = np.argsort(-onnx_output[0])[: len(EXPECTED_TOP_INDICES)]
    if not np.array_equal(top_indices, EXPECTED_TOP_INDICES):
        raise ValueError(f"Unexpected benchmark top indices: {top_indices.tolist()}")
    top_scores = onnx_output[0, top_indices]
    np.testing.assert_allclose(top_scores, EXPECTED_TOP_SCORES, rtol=1e-4, atol=1e-5)

    batch_output = session.run(
        [session.get_outputs()[0].name],
        {session.get_inputs()[0].name: np.vstack([fingerprint, fingerprint])},
    )[0]
    if batch_output.shape != (2, EXPECTED_TEMPLATE_COUNT):
        raise ValueError(
            f"ONNX model does not have a dynamic batch: {batch_output.shape}"
        )
    return [float(score) for score in top_scores]


def convert_reaxys_archive(
    source_mar: Path, output_dir: Path, force: bool = False
) -> Dict[str, Any]:
    """Convert a verified Reaxys MAR into native AiZynthFinder assets."""
    source_mar = source_mar.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not source_mar.is_file():
        raise FileNotFoundError(f"Reaxys archive was not found: {source_mar}")

    source_checksum = _sha256(source_mar)
    if source_checksum != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(
            "Refusing to convert an unknown Reaxys archive: "
            f"expected {EXPECTED_ARCHIVE_SHA256}, got {source_checksum}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = {
        "archive": output_dir / "reaxys.mar",
        "model": output_dir / "reaxys_model.onnx",
        "templates": output_dir / "reaxys_templates.csv.gz",
        "manifest": output_dir / "reaxys_manifest.json",
    }
    existing = [path for path in destinations.values() if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite existing Reaxys assets without --force: "
            + ", ".join(str(path) for path in existing)
        )

    dependencies = _load_conversion_dependencies()
    torch = dependencies[0]
    with tempfile.TemporaryDirectory(
        prefix=".reaxys-convert-", dir=str(output_dir)
    ) as temporary_dir:
        temporary_root = Path(temporary_dir)
        temporary_paths = {
            name: temporary_root / path.name for name, path in destinations.items()
        }
        shutil.copyfile(source_mar, temporary_paths["archive"])

        with zipfile.ZipFile(source_mar) as archive:
            names = set(archive.namelist())
            required_names = {
                "model_latest.pt",
                "templates.jsonl",
                "MAR-INF/MANIFEST.json",
            }
            if not required_names.issubset(names):
                raise ValueError(
                    "Reaxys archive is missing required members: "
                    + ", ".join(sorted(required_names - names))
                )
            template_count = _convert_templates(archive, temporary_paths["templates"])
            checkpoint = _load_checkpoint(torch, archive)

        probability_model = _build_probability_model(torch, checkpoint)
        benchmark_scores = _export_and_validate_model(
            probability_model,
            temporary_paths["model"],
            dependencies,
        )
        manifest = {
            "license": "CC BY-NC 4.0",
            "source_archive": {
                "filename": destinations["archive"].name,
                "sha256": source_checksum,
            },
            "model": {
                "filename": destinations["model"].name,
                "sha256": _sha256(temporary_paths["model"]),
                "format": "ONNX",
                "opset": 13,
                "input_shape": ["batch", EXPECTED_FP_SIZE],
                "output_shape": ["batch", EXPECTED_TEMPLATE_COUNT],
                "output": "softmax probabilities",
                "fingerprint_radius": EXPECTED_RADIUS,
                "fingerprint_chiral": True,
            },
            "templates": {
                "filename": destinations["templates"].name,
                "sha256": _sha256(temporary_paths["templates"]),
                "count": template_count,
                "column": "retro_template",
            },
            "benchmark": {
                "smiles": BENCHMARK_SMILES,
                "top_indices": EXPECTED_TOP_INDICES.tolist(),
                "top_scores": benchmark_scores,
            },
        }
        with temporary_paths["manifest"].open("w", encoding="utf-8") as fileobj:
            json.dump(manifest, fileobj, indent=2, sort_keys=True)
            fileobj.write("\n")

        for name in ("archive", "model", "templates", "manifest"):
            os.replace(temporary_paths[name], destinations[name])
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert the licensed Reaxys MAR to AiZynthFinder assets"
    )
    parser.add_argument("--source-mar", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--force", action="store_true", help="atomically replace existing assets"
    )
    args = parser.parse_args(argv)
    manifest = convert_reaxys_archive(
        args.source_mar, args.output_dir, force=args.force
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
