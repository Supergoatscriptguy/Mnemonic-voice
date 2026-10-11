; audio\ declarations

; wav.asm
extern wav_parse, wav_read, wav_write

; dev.asm
extern dev_open, dev_show, dev_start, dev_stop, dev_close, dev_wait, dev_space, dev_write

; an open wasapi stream
DV_ENUM     equ 0
DV_DEVICE   equ 8
DV_CLIENT   equ 16
DV_SVC      equ 24          ; IAudioRenderClient or IAudioCaptureClient
DV_EVENT    equ 32
DV_FMT      equ 40          ; mix format, dev_close frees it
DV_RATE     equ 48          ; u32
DV_CH       equ 52          ; u32
DV_BUF      equ 56          ; u32, buffer size in frames
DV_DIR      equ 60          ; u32, 0 render 1 capture
DV_PERIOD   equ 64          ; u64, the engine's period in 100 ns units
DV_NAME     equ 72          ; utf-8 friendly name
DV_NAMELEN  equ 128
DV_SIZE     equ 200

; a decoded clip: f32 samples, interleaved, -1..1
CL_DATA     equ 0           ; mem_alloc'd
CL_FRAMES   equ 8           ; u64
CL_RATE     equ 16          ; u32
CL_CH       equ 20          ; u32
CL_BITS     equ 24          ; u32, what the source had
CL_FLOAT    equ 28          ; u32, 1 if the source was float
CL_SIZE     equ 32
