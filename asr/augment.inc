; training data augmentation (asr\augment.asm). include after lib.inc

extern aug_init, aug_utt, aug_spec, aug_out, aug_config

; settings
AG_PSPEED   equ 0               ; f64 chance a micro-batch's speed changes (half 0.9x, half 1.1x)
AG_PRIR     equ 8               ; f64 chance of room reverb
AG_PNOISE   equ 16              ; f64 chance of added noise
AG_FMASK    equ 24              ; u32 frequency masks
AG_FW       equ 28              ; u32 their widest, in bins
AG_TMASK    equ 32              ; u32 time masks
AG_TW       equ 40              ; f64 their widest, a fraction of the frames
AG_POOL     equ 48              ; f64 seconds of noise to keep in memory
AG_SIZE     equ 56

; what aug_utt did, bits of its info word
AI_SLOW     equ 1               ; 0.9x
AI_FAST     equ 2               ; 1.1x
AI_RIR      equ 4
AI_NOISE    equ 8               ; then bits 8-15: the kind, 16-23: the snr in dB
AK_NOISE    equ 0               ; noise kinds
AK_MUSIC    equ 1
AK_BABBLE   equ 2
AK_STREET   equ 3               ; demand: streets, cafes, kitchens, cars...
AK_ROOM     equ 4               ; but reverbdb's rooms, quiet
NKIND       equ 5
