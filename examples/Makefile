TMP := ${TMPDIR}

TOOL := $(shell command -v uv >/dev/null 2>&1 && echo "uvx" || echo "")
RUN := $(shell command -v uv >/dev/null 2>&1 && echo "uv run" || echo "")

all: $(TMP)/sft/ROCO-radiology $(TMP)/sft/ROCOv2-radiology $(TMP)/sft/MedPix-2.0 $(TMP)/sft/MIMIC-CXR

$(TMP)/sft/ROCO-radiology:
	$(RUN) prep sft vqa train --src eltorio/ROCO-radiology --save-dir $(TMP) --no-save-parq
	mv "$(TMP)/sft/vqa" "$(TMP)/sft/ROCO-radiology"

$(TMP)/sft/ROCOv2-radiology:
	$(RUN) prep sft vqa train --src eltorio/ROCOv2-radiology --save-dir $(TMP) --no-save-parq
	mv "$(TMP)/sft/vqa" "$(TMP)/sft/ROCOv2-radiology"

$(TMP)/sft/MedPix-2.0:
	$(RUN) prep sft vqa train --src architojha/medpix-2.0-dataset --save-dir $(TMP) --no-save-parq
	mv "$(TMP)/sft/vqa" "$(TMP)/sft/MedPix-2.0"

$(TMP)/sft/MIMIC-CXR:
	$(RUN) prep sft vqa train --src MLforHealthcare/mimic-cxr --save-dir $(TMP) --no-save-parq --a-cols reports
	mv "$(TMP)/sft/vqa" "$(TMP)/sft/MIMIC-CXR"
