from prep.args import DatasetPrepStream, LoadArgs
from prep.formatter.auto import load_sft
from prep.registry import formatter


@formatter("ROCO-radiology", "sft", "train", default_src="eltorio/ROCO-radiology")
@formatter(
    "ROCOv2-radiology", "sft", "train", default_src="eltorio/ROCOv2-radiology"
)
@formatter(
    "MedPix-2.0", "sft", "train", default_src="architojha/medpix-2.0-dataset"
)
@formatter("MIMIC-CXR", "sft", "train", default_src="MLforHealthcare/mimic-cxr")
@formatter(
    "pixmo-cap-qa", "sft", "train", default_src="anthracite-org/pixmo-cap-qa-images"
)
def load(path: str, split: str, loadargs: LoadArgs) -> "DatasetPrepStream":
    return load_sft(path, split, loadargs)  # ty: ignore[invalid-argument-type]


if __name__ == "__main__":
    # redirect calls that bypass python __file__ to the CLI entrypoint
    import sys

    from typer.main import get_command

    from prep.cli import app

    get_command(app).main(args=sys.argv[1:], standalone_mode=False)
