; inflate.asm
extern inflate_init, inflate_run, inflate_free, gz_open, gz_check, crc32

; a deflate stream being inflated. each inflate_run leaves new output in
; [IF_FROM, IF_N) of the window, the 32 KB before it are history
IF_IN       equ 0           ; compressed bytes (the whole stream, mapped is fine)
IF_INLEN    equ 8
IF_POS      equ 16          ; bit position in IF_IN
IF_OUT      equ 24          ; output window
IF_CAP      equ 32
IF_N        equ 40          ; bytes in the window
IF_FROM     equ 48          ; where this run's output starts
IF_TOTAL    equ 56          ; bytes that slid out of the window before IF_OUT
IF_STATE    equ 64          ; u32: 0 next block, 1 stored, 2 huffman, 3 done
IF_LAST     equ 68          ; u32: this block is the final one
IF_LEFT     equ 72          ; stored bytes still to copy
IF_LBITS    equ 80          ; u32, lit/len table index bits
IF_DBITS    equ 84          ; u32, distance table index bits
IF_LTAB     equ 88          ; -> u16[32768], (symbol << 4) | code length
IF_DTAB     equ 96          ; -> u16[32768]
IF_CRC      equ 104         ; u32, crc32 of everything output so far
IF_LENS     equ 112         ; 320 code lengths for a dynamic block
IF_SIZE     equ 432
