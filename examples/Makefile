RUN ?= $(shell command -v uv >/dev/null 2>&1 && echo "uv run --env-file .env" || echo "")
SAVE_DIR ?= out
SEP ?= ,

.env:
	@echo "" > .env

CSV_FILES := $(wildcard *.csv)
CSV_BASES := $(basename $(CSV_FILES))
nth = $(word $(1),$(subst $(SEP), ,$(2)))

# rows in format,name,split,src
rows_from = $(shell awk -F, '{ sub(/\r$$/, "", $$4); print $$1","$$2","$$3","$$4 }' $(1) 2>/dev/null)

# artifact derivation rule, actual files/directories relative to the SAVE_DIR
# 1/ collective targets, phony name of csv files (without extension)
# Usage: make math       # builds parquet targets from math.csv
#        make math-dirs  # builds directory targets from math.csv
# 2/ individual targets, output path of artifacts specified in the csv file
# Usage: make out/verl/aime24/test.parquet
#        make out/verl/aime24/test/
define PARQ_RULE
$(SAVE_DIR)/$(call nth,1,$(1))/$(call nth,2,$(1))/$(call nth,3,$(1)).parquet: .env
	$(RUN) prep $(call nth,1,$(1)) $(call nth,2,$(1)) $(call nth,3,$(1)) $(call nth,4,$(1)) --save --save-parq --no-hf
endef
define DIR_RULE
$(SAVE_DIR)/$(call nth,1,$(1))/$(call nth,2,$(1))/$(call nth,3,$(1))/: .env
	$(RUN) prep $(call nth,1,$(1)) $(call nth,2,$(1)) $(call nth,3,$(1)) $(call nth,4,$(1)) --save --no-save-parq --no-hf
endef
$(foreach csv,$(CSV_FILES),\
	$(eval _ROWS := $(call rows_from,$(csv))) \
	$(foreach row,$(_ROWS),$(eval $(call PARQ_RULE,$(row)))) \
	$(foreach row,$(_ROWS),$(eval $(call DIR_RULE,$(row)))) \
	$(eval _BASE := $(basename $(csv))) \
	$(eval .PHONY: $(_BASE) $(_BASE)-dirs) \
	$(eval $(_BASE): $(foreach row,$(_ROWS),$(SAVE_DIR)/$(call nth,1,$(row))/$(call nth,2,$(row))/$(call nth,3,$(row)).parquet)) \
	$(eval $(_BASE)-dirs: $(foreach row,$(_ROWS),$(SAVE_DIR)/$(call nth,1,$(row))/$(call nth,2,$(row))/$(call nth,3,$(row))/)) \
)

.PHONY: parq dir help

parq: $(CSV_BASES)
	$(RUN) ppls

dirs: $(CSV_BASES:%=%-dirs)
	$(RUN) ppls

help:
	@echo "\nAvailable targets:"
	@$(eval L1 := $(shell printf '%s\n' $(foreach row,$(ROWS),$(call nth,1,$(row))) | awk '{ len=length; if(len>max) max=len } END { print max }'))
	@$(eval L2 := $(shell printf '%s\n' $(foreach row,$(ROWS),$(call nth,2,$(row))) | awk '{ len=length; if(len>max) max=len } END { print max }'))
	@$(eval L3 := $(shell printf '%s\n' $(foreach row,$(ROWS),$(call nth,3,$(row))) | awk '{ len=length; if(len>max) max=len } END { print max }'))
	@$(eval L4 := $(shell printf '%s\n' $(foreach row,$(ROWS),$(call nth,4,$(row))) | awk '{ len=length; if(len>max) max=len } END { print max }'))
	@$(foreach row,$(ROWS),\
		printf '  %-*s  %-*s  %-*s  %-*s    %-*s\n' $(L1) '$(call nth,1,$(row))' $(L2) '$(call nth,2,$(row))' $(L3) '$(call nth,3,$(row))' $(L4) '$(call nth,4,$(row))'; \
	) :
	@echo ""
