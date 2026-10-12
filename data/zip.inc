; zip.asm
extern zip_open, zip_next, zip_extract

; walking a zip that's in memory
ZP_BASE     equ 0
ZP_LEN      equ 8
ZP_NEXT     equ 16          ; the next central directory entry
ZP_LEFT     equ 24          ; entries still to go
ZP_METHOD   equ 32          ; u32: 0 stored, 8 deflate
ZP_CRC      equ 36          ; u32
ZP_CSIZE    equ 40
ZP_USIZE    equ 48
ZP_DATA     equ 56          ; -> the entry's compressed bytes
ZP_PATH     equ 64
ZP_PATHLEN  equ 512
ZP_SIZEOF   equ ZP_PATH + ZP_PATHLEN
