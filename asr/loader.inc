; the speech data loader: shards -> a table of usable utterances -> a plan of
; micro-batches by length -> filled batches (log-mel, labels). include after lib.inc

extern ld_add, ld_index, ld_plan, ld_fill, ld_norm, ld_scratch

; one loader (a training set, or an eval set)
LD_NSH      equ 0               ; shards open
LD_SH       equ 8               ; -> SR[NSH]
LD_NU       equ 16              ; utterances kept
LD_U        equ 24              ; -> UR[NU], the table
LD_TOK      equ 32              ; -> u16 tokens of every kept utterance, back to back
LD_NTOK     equ 40
LD_NB       equ 48              ; micro-batches in the plan
LD_B        equ 56              ; -> MP[NB]
LD_ORDER    equ 64              ; -> u32[NU]: utterances in plan order, a micro-batch is a run
LD_DROP     equ 72              ; left out: [0] cleaning, [1] too long, [2] too short,
                                ;   [3] too dense for ctc (tokens + repeats > frames at 1.1x)
LD_HOURS    equ 104             ; f64, kept
LD_MAXT     equ 112             ; mel frames an utterance may have (longer ones are left out)
LD_MEAN     equ 120             ; f32[80], the mel normalization (ld_norm)
LD_ISTD     equ 440             ; f32[80], 1 / std
LD_ALL      equ 760             ; 0 = training shards (not dev*, test*, valid*), 1 = every
                                ;   shard, 2 = only the held-out ones
LD_SIZE     equ 768

; a kept utterance
UR_SH       equ 0               ; u16 shard
UR_NT       equ 2               ; u16 tokens
UR_UT       equ 4               ; u32 index in the shard
UR_N        equ 8               ; u32 samples
UR_TOK      equ 12              ; u32 where its tokens start in LD_TOK
UR_SIZE     equ 16

; a planned micro-batch
MP_START    equ 0               ; u32 first entry in LD_ORDER
MP_B        equ 4               ; u32 utterances
MP_T        equ 8               ; u32 mel frames, padded
MP_U        equ 12              ; u32 label row length (its most tokens)
MP_SIZE     equ 16

; a filled batch, host side (pinned, from ld_scratch)
HB_MEL      equ 0               ; f32 [B][T][80], normalized, 0 past each utterance's end
HB_LAB      equ 8               ; u32 [B][U]
HB_TL       equ 16              ; u32 [B] encoder frames, ceil(mel frames / 8)
HB_UL       equ 24              ; u32 [B] labels
HB_B        equ 32              ; u32
HB_T        equ 36              ; u32
HB_U        equ 40              ; u32
HB_NLAB     equ 44              ; u32 labels in all
HB_SIZE     equ 48

; ld_plan flags
PL_FAST     equ 1               ; T a multiple of 128 and B of 8 (B*T/8 a multiple of 128)
