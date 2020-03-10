zip:
	rm -f deepkit.zip
	zip deepkit.zip deepkit/*.py deepkit/utils/*.py README.md setup.cfg setup.py

publish:
	rm -rf dist/*
	python3 setup.py sdist bdist_wheel
	python3 -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*