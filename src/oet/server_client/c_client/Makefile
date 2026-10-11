# Build the compiled oet client. No dependencies beyond libc.
CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra
PREFIX  ?= $(HOME)/.local

oet_client_c: oet_client_c.c
	$(CC) $(CFLAGS) -o $@ $<

install: oet_client_c
	install -d $(PREFIX)/bin
	install -m 0755 oet_client_c $(PREFIX)/bin/oet_client_c

clean:
	rm -f oet_client_c

.PHONY: install clean
