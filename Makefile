
all: docs

.PHONY: all docs clean-docs clean

docs: clean-docs
	sphinx-build -M html docs/ docs/build

clean-docs:
	rm -fr docs/generated/ docs/build/

clean: clean-docs