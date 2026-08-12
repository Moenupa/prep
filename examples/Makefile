TMP := ${TMPDIR}
TMP := ${lst}/datasets

RUN := $(shell command -v uv >/dev/null 2>&1 && echo "uv run" || echo "")

DATASETS := \
    ROCO-radiology \
    ROCOv2-radiology \
    MedPix-2.0 \
    MIMIC-CXR \
    pixmo-cap-qa

TARGETS := $(addprefix $(TMP)/sft/,$(DATASETS))

$(TMP)/sft/%:
	$(RUN) custom.py prep sft $* train --save-dir $(TMP)

.PHONY: all

all: $(TARGETS)
	$(RUN) custom.py ppls $(TMP)
