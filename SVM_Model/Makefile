WORKING_DIR = $(shell pwd)

PYTHON      = ${WORKING_DIR}/.venv/bin/python3
PIP         = ${WORKING_DIR}/.venv/bin/pip3
PACKAGES    = ${WORKING_DIR}/requirements.txt

CORPUS_ALL_FF = ${WORKING_DIR}/work2vec/data/corpus-all.txt


install:
	python -m venv ./.venv
	chmod +x ./.venv/bin/activate
	. ./.venv/bin/activate
	#${PYTHON} --version
	#${PIP} --version
	${PIP} install -r ${PACKAGES}
	cd ${WORKING_DIR}/glove_genvecs && make && cd ${WORKING_DIR}

data:
	. ./.venv/bin/activate
	${PYTHON} -m work2vec.prepare_data
	cp ${CORPUS_ALL_FF} ${WORKING_DIR}/glove_genvecs/text8
	cd ${WORKING_DIR}/glove_genvecs && ./demo.sh && cd ${WORKING_DIR}
	cp -f ${WORKING_DIR}/glove_genvecs/vectors.magnitude ${WORKING_DIR}/work2vec/corpus/

train:
	${PYTHON} -m work2vec.main --train

test:
	. ./.venv/bin/activate
	${PYTHON} -m work2vec.main --test


