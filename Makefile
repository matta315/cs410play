WORKING_DIR = $(shell pwd)
PYTHON      = ${WORKING_DIR}/.venv/bin/python3
PIP         = ${WORKING_DIR}/.venv/bin/pip3

build:
	python3 -m venv ./.venv
	chmod +x ./.venv/bin/activate
	. ./.venv/bin/activate
	#${PYTHON} --version
	#${PIP} --version
	${PYTHON} -m work2vec.main

test:
	echo 'this is just test'


