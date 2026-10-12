; shard files: many utterances, each a whole 16 kHz mono .flac plus its texts.
; the blobs come first, then the index, then a 64 byte trailer (written last, so a
; shard never needs a seek to write). a reader maps the file and starts at the end

; trailer, the file's last 64 bytes
SH_MAGIC    equ 0           ; "MVSHARD1"
SH_COUNT    equ 8           ; u64 utterances
SH_INDEX    equ 16          ; u64 file offset of the index
SH_SAMPLES  equ 24          ; u64 all utterances together
SH_RATE     equ 32          ; u32
SH_SIZE     equ 64

; index entry, one per utterance. offsets are from the start of the file
UT_FLAC     equ 0           ; u64 the .flac
UT_FLEN     equ 8           ; u32
UT_N        equ 12          ; u32 samples
UT_TEXT     equ 16          ; u64 the transcript as written (cased, punctuated), utf-8
UT_TLEN     equ 24          ; u32
UT_SPK      equ 28          ; u32 speaker
UT_NORM     equ 32          ; u64 a normalized transcript, if the source has one
UT_NLEN     equ 40          ; u32
UT_SRC      equ 44          ; u32 which dataset, SRC_*
UT_ID       equ 48          ; u64 the source's own id for it
UT_ILEN     equ 56          ; u32
UT_SIZE     equ 64

; shard.asm
extern shard_open, shard_close
; shardw.asm
extern sw_open, sw_add, sw_close, json_unesc

; a shard being written (sw_open wants it zeroed)
SW_FILE     equ 0
SW_BUF      equ 8
SW_N        equ 16          ; bytes in the buffer
SW_OFF      equ 24          ; the file offset of the buffer's first byte
SW_IDX      equ 32          ; index entries so far
SW_NIDX     equ 40
SW_CAP      equ 48
SW_SAMP     equ 56
SW_PATH     equ 64
SW_TMP      equ 328
SW_FE       equ 592         ; FE (flacenc), 16 bytes
SW_SIZE     equ 608

; one utterance for sw_add
UA_PCM      equ 0           ; int16 samples, 16 kHz mono
UA_N        equ 8
UA_TEXT     equ 16          ; each of these three is (ptr, u32 length)
UA_TLEN     equ 24
UA_NORM     equ 32
UA_NLEN     equ 40
UA_ID       equ 48
UA_ILEN     equ 56
UA_SPK      equ 60
UA_SRC      equ 64
UA_ESC      equ 68          ; 1 = text and norm are json escaped
UA_SIZE     equ 72

; an open shard
SR_BASE     equ 0           ; the mapped file
SR_LEN      equ 8
SR_COUNT    equ 16          ; utterances
SR_INDEX    equ 24          ; -> UT entries
SR_SAMPLES  equ 32
SR_SIZE     equ 40

SRC_HEAVY   equ 1           ; libriheavy (libri-light audio)
SRC_VOXPOP  equ 2           ; voxpopuli
SRC_CV      equ 3           ; common voice
SRC_LS      equ 4           ; librispeech (the test sets)
SRC_NOTGOLD equ 0x100       ; or'd in: voxpopuli says the transcript isn't gold
