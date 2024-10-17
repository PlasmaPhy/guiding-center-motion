
all: docs

.PHONY: all docs clean-docs clean

docs: clean-docs
	sphinx-build -M html ./ ./build

clean-docs:
	rm -fr ./generated/ ./build/

clean: clean-docs