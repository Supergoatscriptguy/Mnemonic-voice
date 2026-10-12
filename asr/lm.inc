; the word n-gram language model (asr\lm.asm): trigrams with stupid backoff, words
; as 64-bit hashes of their lower cased letters. include after lib.inc

extern lm_wstart, lm_wbyte, lm_key2, lm_key3, lm_load, lm_score, lm_find, lm_wtab

LM_MAGIC    equ 'MVLM0001'
LM_START    equ 0x5f5f53544152545f  ; the word before an utterance's first
LM_END      equ 0x5f5f454e445f5f5f  ; the word after its last

; file header, then u64 keys and u32 counts for each table (slots a power of two,
; key 0 = empty)
LH_MAGIC    equ 0
LH_WORDS    equ 8               ; u64 words counted (with the ends)
LH_N1       equ 16              ; u64 slots of each table
LH_N2       equ 24
LH_N3       equ 32
LH_K1       equ 40              ; u64 file offsets of keys and counts
LH_C1       equ 48
LH_K2       equ 56
LH_C2       equ 64
LH_K3       equ 72
LH_C3       equ 80
LH_SIZE     equ 128

; a loaded model
LM_BASE     equ 0               ; the mapped file
LM_K1       equ 8
LM_C1       equ 16
LM_M1       equ 24              ; slots - 1
LM_K2       equ 32
LM_C2       equ 40
LM_M2       equ 48
LM_K3       equ 56
LM_C3       equ 64
LM_M3       equ 72
LM_LOGN     equ 80              ; f64 ln(words)
LM_SIZE     equ 96
