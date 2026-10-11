; audio\ declarations

; wav.asm
extern wav_parse, wav_read, wav_write, clip_mono

; dev.asm
extern dev_open, dev_show, dev_start, dev_stop, dev_close, dev_wait, dev_space, dev_write, dev_read, dev_play

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
DV_PACKETS  equ 200         ; u32, capture packets so far
DV_GLITCH   equ 204         ; u32, capture packets that followed a gap
DV_SILENT   equ 208         ; u32, capture packets windows flagged silent
DV_SIZE     equ 216

; resample.asm
extern rs_init, rs_push, rs_flush, rs_maxout, rs_free, bessel_i0, clip_resample

; a mono resampler
RS_L        equ 0           ; u32, out/in = L/M in lowest terms
RS_M        equ 4           ; u32
RS_H        equ 8           ; u32, taps reach H input samples each side
RS_TAPS     equ 12          ; u32, 2H rounded up to 8
RS_TAB      equ 16          ; L rows of TAPS f32
RS_BUF      equ 24          ; input history + new samples
RS_CAP      equ 32          ; u64, BUF size in samples
RS_HAVE     equ 40          ; u64, samples in BUF
RS_REL      equ 48          ; u64, where the next output's taps start in BUF
RS_PH       equ 56          ; u32, its phase, 0..L-1
RS_NIN      equ 64          ; u64, inputs so far
RS_NOUT     equ 72          ; u64, outputs so far
RS_SIZE     equ 80

; a decoded clip: f32 samples, interleaved, -1..1
CL_DATA     equ 0           ; mem_alloc'd
CL_FRAMES   equ 8           ; u64
CL_RATE     equ 16          ; u32
CL_CH       equ 20          ; u32
CL_BITS     equ 24          ; u32, what the source had
CL_FLOAT    equ 28          ; u32, 1 if the source was float
CL_SIZE     equ 32
