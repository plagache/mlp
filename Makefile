PYTHON_VERSION = 3.12
VENV = .venv
BIN = ${VENV}/bin
PYTHON = ${BIN}/python
ACTIVATE = ${BIN}/activate

EXAMPLES = examples

PROGRAM = mlp.py

# ARGUMENTS =

setup: venv pip_upgrade install

venv:
	uv venv --python ${PYTHON_VERSION} ${VENV} --seed
	ln -sf ${ACTIVATE} activate

pip_upgrade:
	uv pip install --upgrade pip

install: \
	requirements \
	# module \
#
requirements: requirements.txt
	uv pip install -r requirements.txt --upgrade

module: setup.py
	uv pip install -e . --upgrade

list:
	uv pip list

version:
	uv python list

size:
	du -hd 0
	du -hd 0 ${VENV}

run:
	PYTHONPATH=. ${PYTHON} ${EXAMPLES}/${PROGRAM} \
	# ${ARGUMENTS}

test:
	PYTHONPATH=. ${PYTHON} ${EXAMPLES}/test.py

tiny:
	CPU=1 PYTHONPATH=. ${PYTHON} ${EXAMPLES}/tiny.py

clean:

fclean: clean
	rm -rf ${VENV}
	rm -rf activate

re: fclean setup run

.SILENT:
.PHONY: setup venv pip_upgrade install module requirements list version run clean fclean re
