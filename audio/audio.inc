; audio\ declarations

; wav.asm
extern wav_parse, wav_head, wav_read, wav_write, clip_mono
extern level_db

; flac.asm
extern flac_info, flac_decode, md5

; a flac stream's header
FI_RATE     equ 0
FI_CH       equ 4
FI_BITS     equ 8
FI_MINBS    equ 12
FI_MAXBS    equ 16
FI_MINFR    equ 20
FI_MAXFR    equ 24
FI_BLOCKS   equ 28          ; bit t set = there's a metadata block of type t
FI_SAMPLES  equ 32          ; per channel, 0 = unknown
FI_MD5      equ 40          ; 16 bytes, all 0 = not set
FI_FRAMES   equ 56          ; -> the first frame
FI_SIZE     equ 64

; flacenc.asm. flac_put is the bit writer (rdi = next byte, r14 = pending bits,
; r15d = how many), for building test streams by hand
extern flac_encode, flac_put, flac_crc8, flac_crc16

FE_DATA     equ 0           ; -> the .flac, mem_alloc'd
FE_LEN      equ 8
FE_SIZE     equ 16

; a decoded flac stream: the FI above, then
FD_PCM      equ FI_SIZE         ; -> int16 samples, interleaved, mem_alloc'd
FD_N        equ FI_SIZE + 8     ; samples a channel that got decoded
FD_MD5      equ FI_SIZE + 16    ; u32: 1 the md5 matched, 0 it didn't, 2 the stream has none
FD_STATS    equ FI_SIZE + 24    ; u64 counts: constant, verbatim, fixed order 0-4, lpc,
                                ; rice2, escapes, wasted bits, frames, highest lpc order
FD_SIZE     equ FI_SIZE + 24 + 128

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

; mel.asm
extern mel_init, mel_power, mel_apply, mel_run, hz2mel, mel2hz

MEL_N       equ 80          ; filters
MEL_WINLEN  equ 400         ; 25 ms
MEL_HOP     equ 160         ; 10 ms
; the tables mel_init fills
MEL_WIN     equ 0           ; f32[400] periodic hann
MEL_TWC     equ 1600        ; f32[128] cos, 256 point fft twiddles
MEL_TWS     equ 2112        ; f32[128] sin
MEL_RC      equ 2624        ; f32[257] cos(2 pi k / 512) for the real split
MEL_RS      equ 3664        ; f32[257] sin
MEL_REV     equ 4704        ; u16[256] 8 bit reversal
MEL_FB0     equ 5216        ; u16[80] each filter's first bin
MEL_FBN     equ 5376        ; u16[80] how many bins it has
MEL_FBW     equ 5536        ; u32[80] where its weights start in MEL_W
MEL_W       equ 5856        ; f32 weights, all filters back to back
MEL_SIZE    equ MEL_W + 4096

; a decoded clip: f32 samples, interleaved, -1..1
CL_DATA     equ 0           ; mem_alloc'd
CL_FRAMES   equ 8           ; u64
CL_RATE     equ 16          ; u32
CL_CH       equ 20          ; u32
CL_BITS     equ 24          ; u32, what the source had
CL_FLOAT    equ 28          ; u32, 1 if the source was float
CL_SIZE     equ 32

extern mp3_head, mp3_info, mp3_init, mp3_decode
M3_SIZE     equ 16384 + 4096 + 1024  ; an mp3_decode state (the reservoir reads can overrun)

; what an mp3's frame headers say (mp3_info)
MI_VER      equ 0           ; u32, 1 / 2 / 25 for mpeg-1 / 2 / 2.5, from the first frame
MI_RATE     equ 4           ; u32
MI_MODE     equ 8           ; u32, 0 stereo 1 joint 2 dual 3 mono
MI_MODES    equ 12          ; u32, every mode seen, a bit each
MI_MINBR    equ 16          ; u32 kbps
MI_MAXBR    equ 20          ; u32
MI_FRAMES   equ 24          ; u64
MI_SAMPLES  equ 32          ; u64 a channel
MI_SKIP     equ 40          ; u64 bytes skipped looking for a frame
MI_TAIL     equ 48          ; u64 bytes after the last whole frame
MI_XING     equ 56          ; u32, the first frame is a xing/info header
MI_MIXED    equ 60          ; u32, the rate or version changes
MI_CRC      equ 64          ; u32, frames carry a crc
MI_SIZE     equ 72
