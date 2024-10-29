CONFIGS_DIR ?= configs
TARGETS_DIR ?= infos
CONFIGS :=  $(wildcard $(CONFIGS_DIR)/*.yaml)
TARGETS := $(patsubst configs/%, $(TARGETS_DIR)/%, $(CONFIGS:.yaml=.pkl))

all: $(TARGETS)

$(TARGETS_DIR)/%.pkl: $(CONFIGS_DIR)/%.yaml
	poetry run python fkdc/process.py --config-file $^ --workdir $(TARGETS_DIR)

datasets: fkdc/datasets.py
	poetry run python $^

configs: fkdc/config.py
	poetry run python $^

clean:
	rm -f $(TARGETS)

.PHONY: all clean