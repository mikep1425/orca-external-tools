# Build the compiled oet client. No dependencies beyond libc.
CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra
PREFIX  ?= $(HOME)/.local

oet_client_c: oet_client.c
	$(CC) $(CFLAGS) -o $@ $<

install: oet_client_c
	install -d $(PREFIX)/bin
	install -m 0755 oet_client_c $(PREFIX)/bin/oet_client_c

# Saves ~0.3 ms/call of dynamic linking. Use only with numeric hosts
# (-b 127.0.0.1:PORT): statically linked name resolution still needs glibc's
# NSS shared objects at runtime.
static: oet_client.c
	$(CC) $(CFLAGS) -static -o oet_client_c $<

clean:
	rm -f oet_client_c

.PHONY: install static clean
