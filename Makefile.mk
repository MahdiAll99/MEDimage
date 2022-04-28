#
# Configuration variables
#

WORKDIR?=./MEDimage
REQUIREMENTS_TXT?=environment.yml
SETUP_PY?=setup.py
python_version := $(wordlist 2,4,$(subst ., ,$(shell python --version 2>&1)))

#
# Virtual environment
#

.PHONY: create_environment
create_environment:
	conda update --yes --name base --channel defaults conda
	conda env create --file environment.yml

.PHONY: clean
clean:
	find . -type f -name *.pyc -delete
	find . -type d -name __pycache__ -delete

.PHONY: debug_env
debug_env:
	@$(MAKE) --version
	$(info Python="$(python_version)")
	$(info REQUIREMENTS="$(REQUIREMENTS_TXT)")
	$(info SETUP_PY="$(SETUP_PY)")
	$(info WORKDIR="$(WORKDIR)")
