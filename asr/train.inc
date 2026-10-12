; the speech trainer's checkpoints (asr\ckpt.asm)

extern ck_save, ck_load, ck_latest, ck_prune

; checkpoint file: this header, then the f32 master weights, adam m, adam v
CK_MAGIC_V equ 'MVCKPT01'
CK_MAGIC   equ 0
CK_L       equ 8                ; the model's shape, has to match on load
CK_D       equ 16
CK_H       equ 24
CK_F       equ 32
CK_V       equ 40
CK_K       equ 48
CK_C0      equ 56
CK_NMEL    equ 64
CK_NP      equ 72
CK_STEP    equ 80               ; steps done
CK_SEEN    equ 88               ; f64 hours of audio trained on
CK_EPOCH   equ 96               ; which epoch's plan, and where in it
CK_POS     equ 104
CK_ELAPSED equ 112              ; f64 seconds, over all sessions
CK_VLOSS   equ 120              ; f64, latest val loss
CK_TOTAL   equ 128              ; steps planned (just for the record)
CK_MEAN    equ 256              ; the mel normalization it was trained with, f32[80]
CK_ISTD    equ 576              ;   and 1/std, f32[80]
CK_SIZE    equ 4096             ; the weights start on a page
