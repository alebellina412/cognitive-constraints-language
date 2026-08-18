CC := gcc
CFLAGS := -O2 -std=c11 -D_DEFAULT_SOURCE -Wall -Wextra
LDLIBS := -lm

.PHONY: all clean

all: tests/umt_legacy tests/umt_zstd tests/umt_fast_zstd

tests/umt_legacy: legacy/UMT_dynamic.c legacy/polya_adj.h
	$(CC) $(CFLAGS) -o $@ legacy/UMT_dynamic.c $(LDLIBS)

tests/umt_zstd: compressed/UMT_dynamic_zstd.c legacy/UMT_dynamic.c legacy/polya_adj.h
	$(CC) $(CFLAGS) -o $@ compressed/UMT_dynamic_zstd.c $(LDLIBS)

tests/umt_fast_zstd: optimized/UMT_dynamic_fast_zstd.c
	$(CC) $(CFLAGS) -o $@ optimized/UMT_dynamic_fast_zstd.c $(LDLIBS)

clean:
	rm -f tests/umt_legacy tests/umt_zstd tests/umt_fast_zstd
