; audio\ declarations

; wav.asm
extern wav_parse, wav_read, wav_write

; a decoded clip: f32 samples, interleaved, -1..1
CL_DATA     equ 0           ; mem_alloc'd
CL_FRAMES   equ 8           ; u64
CL_RATE     equ 16          ; u32
CL_CH       equ 20          ; u32
CL_BITS     equ 24          ; u32, what the source had
CL_FLOAT    equ 28          ; u32, 1 if the source was float
CL_SIZE     equ 32
