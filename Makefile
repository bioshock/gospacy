.PHONY: help test test-blis lint fmt diff-test diff-test-ops bootstrap-ref download-assets build clean

help:
	@echo "gospacy — make targets:"
	@echo "  bootstrap-ref     create Python ref venv and install requirements-ref.txt"
	@echo "  download-assets   fetch reference model and corpora into testdata/"
	@echo "  diff-test         regenerate ALL golden fixtures (pipeline + ops + murmur)"
	@echo "  diff-test-ops     regenerate ops + murmur golden fixtures only"
	@echo "  test              go test ./... (pure-Go default)"
	@echo "  test-blis         go test -tags blis ./..."
	@echo "  lint              golangci-lint run"
	@echo "  fmt               gofmt -s -w ."
	@echo "  build             go build ./..."
	@echo "  clean             remove generated artefacts"

test:
	go test ./...

test-blis:
	go test -tags blis ./...

lint:
	golangci-lint run

fmt:
	gofmt -s -w .
	go vet ./...

build:
	go build ./...

bootstrap-ref:
	bash testharness/bootstrap.sh

download-assets:
	bash testharness/download_assets.sh

diff-test:
	testharness/.venv/bin/python testharness/dump_all.py
	testharness/.venv/bin/python testharness/dump_ops.py all
	testharness/.venv/bin/python testharness/dump_murmur.py
	testharness/.venv/bin/python testharness/dump_stringstore.py
	testharness/.venv/bin/python testharness/dump_lex_attrs.py
	testharness/.venv/bin/python testharness/dump_tokenizer.py
	testharness/.venv/bin/python testharness/dump_bundle.py
	testharness/.venv/bin/python testharness/dump_tagger.py
	testharness/.venv/bin/python testharness/dump_attribute_ruler.py
	testharness/.venv/bin/python testharness/dump_lemmatizer.py

diff-test-ops:
	testharness/.venv/bin/python testharness/dump_ops.py all
	testharness/.venv/bin/python testharness/dump_murmur.py

clean:
	rm -rf testdata/models testdata/corpora
	rm -rf testharness/.venv
	rm -rf coverage.txt dist/
