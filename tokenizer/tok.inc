; tokenizer: constants and shared declarations. include after lib.inc.
; Mnemonic's byte-level bpe (pretok, bpe, tok are its files, unchanged), sized for
; speech: 1024 outputs = 256 bytes + 767 merges + the ctc blank

extern pretok_next, whash, memeq
extern bpe_train
extern tok_load, tok_encode, tok_encode_chunk, tok_decode, tok_bytes
extern tok_nmerges, tok_hash, tok_cache_new

VOCAB      equ 1024
NSPECIAL   equ 1
SPECIAL0   equ VOCAB - NSPECIAL ; the specials sit at the top
TOK_BLANK  equ SPECIAL0         ; the ctc blank, never in any text
NMERGES    equ SPECIAL0 - 256   ; 767
MAXWORD    equ 256              ; longest chunk that bpe training looks at
MASKBIT    equ 0x8000           ; tok.asm masks it off (Mnemonic's chat targets); unused here.
                                ; tok_decode would name the blank "<|bos|>", but ctc drops blanks first

; encode context (tok_cache_new), one per thread
TC_CACHE   equ 0
TC_WORK    equ 8
TC_NOCACHE equ 16               ; set to 1 to skip the cache (for checking it)
TC_SIZE    equ 64

; tokenizer file: header, then u32 merges (a | b << 16), merge i makes token 256+i
TK_MAGIC     equ 'MVSPTOK1'
TKH_MAGIC    equ 0
TKH_VOCAB    equ 8              ; u32
TKH_NMERGES  equ 12             ; u32
TKH_SPECIAL0 equ 16             ; u32
TKH_NSPECIAL equ 20             ; u32
TKH_HASH     equ 24             ; u64, of the merges
TKH_SIZE     equ 32

; bpe_train
BT_DOCS    equ 0                ; -> (ptr, len) per doc
BT_NDOCS   equ 8
BT_NMERGES equ 16
BT_NAIVE   equ 24               ; 1 = recount every pair for every merge (slow, for checking)
BT_VERBOSE equ 32
BT_MERGES  equ 40               ; -> u32 per merge, a | b << 16 (filled in)
BT_COUNTS  equ 48               ; -> u64 per merge, how often that pair occurred
BT_DONE    equ 56               ; merges actually done
BT_WORDS   equ 64               ; unique words seen
BT_BAD     equ 72               ; verification problems, should be 0
BT_SIZE    equ 80
