from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from datasets import ClassLabel, Features, Image, List, Value

from ..constants import ERROR_PREFIX, ID_PATTERN, IMAGE_TAG, WARN_PREFIX
from .log import get_logger
from .types import (
    DataFormat,
    ProcArgs,
    RegistrationError,
    Split,
    get_valid_splits,
)
from .validate import (
    count_img_tags,
    validate_answer_formatting,
    validate_image_tags,
    validate_openai_format,
)

if TYPE_CHECKING:
    from datasets import Dataset

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
        if not ID_PATTERN.match(id_):
            raise RegistrationError(
                f"Invalid pipeline ID {id_!r}. Must match {ID_PATTERN.pattern!r}"
            )
        if target_format == "show" and id_ != "_":
            raise RegistrationError(
                "No new pipelines allowed for target_format 'show'."
            )

        if (id_, target_format, split) in _FORMATTER_REGISTRY:
            raise RegistrationError(
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
            logger.warning(f"{WARN_PREFIX}Formatter pipeline not registered for {k}. ")
            logger.warning(
                "Fallback to generic pipeline 'auto',"
                " which may cause unexpected formatting issues.",
            )
            return _FORMATTER_REGISTRY[("auto", target_format, split)]

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
                validate_answer_formatting(sample["prompt"])
            case "sft":
                validate_openai_format(sample["messages"])
                validate_image_tags(sample["messages"], expected_n_img=n_img)
            case "eval":
                n_tags = count_img_tags(sample["question"])
                if n_tags != n_img:
                    raise ValueError(
                        f"Mismatch: number of images {n_img} != {n_tags} {IMAGE_TAG} tags."
                    )

    @staticmethod
    def cast_verl(d: "Dataset", args: ProcArgs) -> "Dataset":
        return d.cast(
            Features(
                images=List(Image(decode=True)),
                data_source=Value("string"),
                prompt=List(
                    {"role": Value("string"), "content": Value("large_string")}
                ),
                ability=Value("string"),
                reward_model={
                    "style": Value("string"),
                    "ground_truth": Value("string"),
                },
                extra_info={
                    "split": Value("string"),
                    "index": Value("string"),
                    # feedback, CoT, or hint to guide better answers
                    "explanation": Value("large_string"),
                    # any miscellaneous info accepting json.dumps() stuff
                    # this is for compatiblity with multiple datasets, supporting any structure
                    "misc": Value("large_string"),
                },
            ),
            num_proc=args.num_proc,
        )

    @staticmethod
    def cast_sft(d: "Dataset", args: ProcArgs) -> "Dataset":
        return d.cast(
            Features(
                images=List(Image(decode=True)),
                messages=List(
                    {"role": Value("string"), "content": Value("large_string")}
                ),
                id=Value("string"),
                extra_info=Value("large_string"),
            ),
            num_proc=args.num_proc,
        )

    @staticmethod
    def cast_eval(d: "Dataset", args: ProcArgs) -> "Dataset":
        return d.cast(
            Features(
                id=Value("string"),
                images=List(Image(decode=True)),
                question=Value("string"),
                options=List(Value("string")),
                answer=Value("string"),
            ),
            num_proc=args.num_proc,
        )

    @staticmethod
    def cast_cls(d: "Dataset", args: ProcArgs) -> "Dataset":
        if isinstance(d.features.get("label"), ClassLabel):
            label_feature = d.features["label"]
        elif args.labels:
            label_feature = ClassLabel(names=args.labels)
        else:
            raise ValueError(
                "Please provide labels via ENV `LABELS='cls1 cls2'`"
                " or CLI option `--labels cls1 --labels cls2`."
            )

        return d.cast(
            Features(
                id=Value("string"),
                image=Image(decode=True),
                label=label_feature,
                extra_info=Value("large_string"),
            ),
            num_proc=args.num_proc,
        )

    @staticmethod
    def cast_dataset(
        d: "Dataset", target_format: DataFormat, args: ProcArgs
    ) -> "Dataset":
        match target_format:
            case "verl":
                return FormatterPipeline.cast_verl(d, args)
            case "sft":
                return FormatterPipeline.cast_sft(d, args)
            case "eval":
                return FormatterPipeline.cast_eval(d, args)
            case "cls":
                return FormatterPipeline.cast_cls(d, args)
            case "show":
                return d
            case _:
                raise ValueError(f"Unknown target_format {target_format!r}")

    def load(self, override_src: str | None, args: ProcArgs) -> "Dataset":
        path = override_src or self.default_src
        if path is None:
            raise ValueError(
                f"Dataset {self.id_!r} has no local/remote source to load from."
                " Pass a source (e.g. `SRC=`) to override or register a default source."
            )
        d = self.load_fn(path, self.split, args)
        try:
            d = self.cast_dataset(d, self.target_format, args)
        except Warning as e:
            logger.warning(f"{WARN_PREFIX}Casting warning {str(self)}")
            logger.warning(f"{WARN_PREFIX}{e}")
        except Exception as e:
            logger.error(f"{ERROR_PREFIX}Casting failed {str(self)}")
            logger.error(f"{ERROR_PREFIX}{e}")
        try:
            for i in range(10):
                self.check_sample(d[i])
        except Warning as e:
            logger.warning(f"{WARN_PREFIX}Validation warning {str(self)}")
            logger.warning(f"{WARN_PREFIX}{e}")
        except Exception as e:
            logger.error(f"{ERROR_PREFIX}Validation failed {str(self)}")
            logger.error(f"{ERROR_PREFIX}{e}")

        return d


def get_registered_pipelines() -> dict[tuple[DataFormat, str], str | None]:
    return {
        (pipeline.target_format, pipeline.id_): pipeline.default_src
        for pipeline in _FORMATTER_REGISTRY.values()
    }
