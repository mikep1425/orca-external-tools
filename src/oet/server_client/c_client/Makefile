# Build the compiled oet client. No dependencies beyond libc.
CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra
PREFIX  ?= $(HOME)/.local

oet_client: oet_client.c
	$(CC) $(CFLAGS) -o $@ $<

install: oet_client
	install -d $(PREFIX)/bin
	install -m 0755 oet_client $(PREFIX)/bin/oet_client

clean:
	rm -f oet_client

.PHONY: install clean
