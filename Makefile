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
	cd ${WORKING_DIR}/glove_genvecs && make && cd ${WORKING_DIR}

# Run using pre-trained SVM
run:
	. ./.venv/bin/activate
	${PYTHON} -m work2vec.main

train:
	. ./.venv/bin/activate
	${PYTHON} -m work2vec.gen_dict_for_glove
	cp work2vec/corpus-all.txt glove_genvecs/text8
	cd ${WORKING_DIR}/glove_genvecs && ./demo.sh && cd ${WORKING_DIR}
	cp -f ${WORKING_DIR}/glove_genvecs/vectors.magnitude ${WORKING_DIR}/work2vec/
	${PYTHON} -m work2vec.main

test:
	echo 'just test'


