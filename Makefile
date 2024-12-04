lint:
	pylint --recursive=y code
	flake8 code

isort:
	isort code

requirements:
	pip-compile requirements.in

install-requirements:
	pip install --no-cache-dir -r requirements.txt

test:
	pytest
