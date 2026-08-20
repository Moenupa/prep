RUN := $(shell command -v uv >/dev/null 2>&1 && echo "uv run" || echo "")

SAVE_DIR ?= out
CSV_FILES := $(wildcard *.csv)

# Parse each CSV as four non-empty comma-separated fields.
# Target components must contain only letters, numbers, ".", "_", "-", and "/".
ifneq ($(strip $(CSV_FILES)),)
CSV_ROWS_RAW := $(shell \
	awk -F, '\
		function fail(message) { \
			printf "%s:%d: %s\n", FILENAME, FNR, message > "/dev/stderr"; \
			bad = 1 \
		} \
		{ \
			sub(/\r$$/, "", $$4); \
			if (NF != 4) { \
				fail("expected exactly 4 columns"); \
				next \
			} \
			for (i = 1; i <= 4; i++) \
				if ($$i == "" || $$i ~ /[[:space:]|]/ || \
					index($$i, sprintf("%c", 39)) || index($$i, "$$")) { \
					fail("empty or unsupported field"); \
					next \
				} \
			if ($$1 !~ /^[[:alnum:]_.\/-]+$$/ || \
				$$2 !~ /^[[:alnum:]_.\/-]+$$/ || \
				$$3 !~ /^[[:alnum:]_.\/-]+$$/) { \
				fail("invalid character in target component"); \
				next \
			} \
			printf "%s|%s|%s|%s\n", $$1, $$2, $$3, $$4 \
		} \
		END { if (bad) exit 1 }' $(CSV_FILES) || \
	echo __CSV_PARSE_ERROR__)
else
CSV_ROWS_RAW :=
endif

ifneq ($(filter __CSV_PARSE_ERROR__,$(CSV_ROWS_RAW)),)
$(error Could not parse CSV input)
endif

CSV_ROWS := $(CSV_ROWS_RAW)
csv_field = $(word $(1),$(subst |, ,$(2)))

REAL_TARGETS := $(foreach row,$(CSV_ROWS),\
	$(SAVE_DIR)/$(call csv_field,1,$(row))/$(call csv_field,2,$(row))/$(call csv_field,3,$(row)).parquet)

ifneq ($(words $(REAL_TARGETS)),$(words $(sort $(REAL_TARGETS))))
$(error Multiple CSV rows produce the same target)
endif

.PHONY: all help

all: $(REAL_TARGETS)
	$(RUN) ppls

help:
	@echo "\nAvailable targets:"
	@$(eval F1_MAX := $(shell printf '%s\n' $(foreach row,$(CSV_ROWS),$(call csv_field,1,$(row))) | awk '{ len=length; if(len>max) max=len } END { print max }'))
	@$(eval F2_MAX := $(shell printf '%s\n' $(foreach row,$(CSV_ROWS),$(call csv_field,2,$(row))) | awk '{ len=length; if(len>max) max=len } END { print max }'))
	@$(eval F3_MAX := $(shell printf '%s\n' $(foreach row,$(CSV_ROWS),$(call csv_field,3,$(row))) | awk '{ len=length; if(len>max) max=len } END { print max }'))
	@$(eval TGT_MAX := $(shell printf '%s\n' $(foreach row,$(CSV_ROWS),$(SAVE_DIR)/$(call csv_field,1,$(row))/$(call csv_field,2,$(row))/$(call csv_field,3,$(row)).parquet) | awk '{ len=length; if(len>max) max=len } END { print max }'))
	@$(eval SRC_MAX := $(shell printf '%s\n' $(foreach row,$(CSV_ROWS),$(call csv_field,4,$(row))) | awk '{ len=length; if(len>max) max=len } END { print max }'))
	@$(foreach row,$(CSV_ROWS),\
		printf '  %-*s  %-*s  %-*s  %-*s    %-*s\n' $(F1_MAX) '$(call csv_field,1,$(row))' $(F2_MAX) '$(call csv_field,2,$(row))' $(F3_MAX) '$(call csv_field,3,$(row))' $(TGT_MAX) '$(SAVE_DIR)/$(call csv_field,1,$(row))/$(call csv_field,2,$(row))/$(call csv_field,3,$(row)).parquet' $(SRC_MAX) '$(call csv_field,4,$(row))'; \
	) :
	@echo ""

define CSV_RULE
$(SAVE_DIR)/$(call csv_field,1,$(1))/$(call csv_field,2,$(1))/$(call csv_field,3,$(1)).parquet:
	$(RUN) prep '$(call csv_field,1,$(1))' '$(call csv_field,2,$(1))' '$(call csv_field,3,$(1))' --src '$(call csv_field,4,$(1))' --save --save-parq --no-hf
endef

$(foreach row,$(CSV_ROWS),$(eval $(call CSV_RULE,$(row))))
