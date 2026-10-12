; tar.asm
extern tar_open, tar_next

; walking a tar that's in memory
TA_POS      equ 0           ; next header
TA_END      equ 8
TA_LONG     equ 16          ; a gnu long name waiting for its header, or 0
TA_NAME     equ 24          ; -> this file's name
TA_DATA     equ 32          ; -> its bytes
TA_SIZE     equ 40
TA_PATH     equ 48
TA_PATHLEN  equ 1024
TA_SIZEOF   equ TA_PATH + TA_PATHLEN
