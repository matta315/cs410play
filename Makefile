WORKING_DIR = $(shell pwd)
PYTHON      = ${WORKING_DIR}/.venv/bin/python3
PIP         = ${WORKING_DIR}/.venv/bin/pip3
PACKAGES    = ${WORKING_DIR}/requirements.txt

install:
	python3 -m venv ./.venv
	chmod +x ./.venv/bin/activate
	. ./.venv/bin/activate
	#${PYTHON} --version
	#${PIP} --version
	${PIP} install -r ${PACKAGES}

# Run using pre-trained SVM
run:
	. ./.venv/bin/activate
	${PYTHON} -m work2vec.main

train:
	echo 'train'

test:
	echo 'just test'


