from collections.abc import Callable
from dataclasses import dataclass
from sys import stderr

from datasets import Dataset

from ..constants import IMAGE_TAG, WARN_PREFIX
from .log import get_logger
from .types import (
    EVAL_FEAT,
    SFT_FEAT,
    VERL_FEAT,
    DataFormat,
    ProcArgs,
    Split,
    get_valid_splits,
)
from .validate import count_img_tags, validate_image_tags, validate_openai_format

logger = get_logger(__name__)
type LoadFn = Callable[[str, Split, ProcArgs], "Dataset"]

_FORMATTER_REGISTRY: dict[tuple[str, DataFormat, Split], "FormatterPipeline"] = {}


def formatter(
    id_: str,
    target_format: DataFormat,
    split: Split,
    default_src: str | None = None,
) -> Callable[[LoadFn], LoadFn]:
    def decorator(function: LoadFn) -> LoadFn:
        assert "/" not in id_, "pipeline ID should not contain '/'"
        assert (id_, target_format, split) not in _FORMATTER_REGISTRY, (
            f"Pipeline duplicated for {(id_, target_format, split)}"
        )

        _FORMATTER_REGISTRY[(id_, target_format, split)] = FormatterPipeline(
            id_=id_,
            target_format=target_format,
            split=split,
            load_fn=function,
            default_src=default_src,
        )
        logger.debug(
            f"🗂\tRegistered {id_!r} (fmt={target_format!r}, split={split!r}, src={default_src!r})"
        )
        return function

    return decorator


@dataclass(frozen=True)
class FormatterPipeline:
    id_: str
    target_format: DataFormat
    split: Split

    load_fn: LoadFn
    default_src: str | None = None

    def __str__(self) -> str:
        return (
            f"Formatter(id={self.id_!r}, target_format={self.target_format!r}"
            f", split={self.split!r}, default_src={self.default_src!r})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def get(
        cls, id_: str, target_format: DataFormat, split: Split
    ) -> "FormatterPipeline":
        if target_format == "show":
            return _FORMATTER_REGISTRY[("_", "show", split)]
        k = (id_, target_format, split)
        if k not in _FORMATTER_REGISTRY:
            print(
                f"{WARN_PREFIX}Formatter pipeline not registered for {k}. "
                "Fallback to the general pipeline 'vqa', which may result in expected formatting issues.",
                file=stderr,
            )
            return _FORMATTER_REGISTRY[("vqa", target_format, split)]

        return _FORMATTER_REGISTRY[(id_, target_format, split)]

    @staticmethod
    def unregistered_splits(id_: str, target_format: DataFormat) -> dict[Split, None]:
        return {
            split: None
            for split in get_valid_splits()
            if (id_, target_format, split) not in _FORMATTER_REGISTRY
        }

    def check_sample(self, sample: dict) -> None:
        n_img = len(sample.get("images", []))
        match self.target_format:
            case "verl":
                validate_openai_format(sample["prompt"])
                validate_image_tags(sample["prompt"], expected_n_img=n_img)
            case "sft":
                validate_openai_format(sample["messages"])
                validate_image_tags(sample["messages"], expected_n_img=n_img)
            case "eval":
                n_tags = count_img_tags(sample["question"])
                assert n_tags == n_img, (
                    f"Mismatch: number of images {n_img} != {n_tags} {IMAGE_TAG} tags."
                )

    def load(self, override_src: str | None, args: ProcArgs) -> Dataset:
        path = override_src or self.default_src
        if path is None:
            raise ValueError(
                f"Dataset {self.id_!r} has no local/remote source to load from."
                " Pass --src to override or register a default source in loading function."
            )
        d = self.load_fn(path, self.split, args)
        try:
            match self.target_format:
                case "verl":
                    d = d.cast(VERL_FEAT)
                case "sft":
                    d = d.cast(SFT_FEAT)
                case "eval":
                    d = d.cast(EVAL_FEAT)
        except Exception as e:
            logger.error(f"⚠️\tCasting failed {str(self)}\n⚠️\t{e}")
        try:
            for i in range(10):
                self.check_sample(d[i])
        except Exception as e:
            logger.error(f"⚠️\tValidation failed {str(self)}\n⚠️\t{e}")

        args.peek(d)
        return d


def get_registered_pipelines() -> dict[tuple[DataFormat, str], str | None]:
    return {
        (pipeline.target_format, pipeline.id_): pipeline.default_src
        for pipeline in _FORMATTER_REGISTRY.values()
    }
