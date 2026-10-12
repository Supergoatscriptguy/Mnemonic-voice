; the speech encoder: a conformer over 8x subsampled log-mel frames, with a ctc
; head. config, sizes, buffers. include after lib.inc and gpu/cuda.inc

extern model_config, model_setup, model_init, model_cast, model_fwd, model_bwd, model_adam0
extern model_zero, model_step, model_summary, model_argmax
extern mdl, mdl_fast, lp, fe, d_params, d_bf, d_grad, d_adm, d_adv, d_logits, d_gn

; mdl: the config plus everything derived from it. all u64
MD_L      equ 0                 ; conformer blocks
MD_D      equ 8
MD_H      equ 16                ; heads (as many kv heads)
MD_HD     equ 24                ; D / H
MD_F      equ 32                ; feed-forward hidden size: gate and up are F each
MD_V      equ 40                ; vocab, the blank last
MD_K      equ 48                ; the conv module's kernel
MD_C0     equ 56                ; subsampling channels
MD_NMEL   equ 64                ; mel bins
MD_T      equ 72                ; mel frames per utterance at most (activations are sized for it)
MD_B      equ 80                ; utterances per micro-batch at most
MD_UMAX   equ 88                ; labels per utterance at most
MD_F1     equ 96                ; mel bins after each stride-2 conv
MD_F2     equ 104
MD_F3     equ 112
MD_T3     equ 120               ; encoder frames per utterance at most, MD_T / 8 rounded up
MD_M      equ 128               ; B * T3, the encoder's rows at most
MD_QKV    equ 136               ; 3 D
MD_ES     equ 144               ; bytes per activation: 2 (bf16) or 4 (f32, for gradient checks)
MD_NP     equ 152               ; params
MD_NDEC   equ 160               ; params [0, NDEC) are matrices, they get weight decay
MD_LSIZE  equ 168               ; matrix params per block
MD_ACTL   equ 176               ; activation bytes per block
MD_ROPE   equ 184               ; rope base (f64)
MD_FLOPS  equ 192               ; training flops per encoder frame at full length (f64)
MD_SIZE   equ 200

; per block pointers, lp + l*LP_SIZE. entry L is the final norm (X, XNA, RSA, NFA, GNFA)
; a block: x1 = x  + ff(norm(x))                   (swiglu, the macaron's first half)
;          x2 = x1 + attention(norm(x1))           (rope, the chunk/length mask)
;          x3 = x2 + conv(norm(x2))                (pointwise, glu, causal depthwise,
;                                                   norm, silu, pointwise)
;          x4 = x3 + ff(norm(x3))                  (the second half), the next block's x
LP_W13A   equ 0                 ; matrices as the matmuls see them: ff1 gate+up [2F][D]
LP_W2A    equ 8                 ;   ff1 down [D][F]
LP_WQKV   equ 16                ;   [3D][D]
LP_WO     equ 24                ;   [D][D]
LP_WPW1   equ 32                ;   conv module in [2D][D] (glu's two halves)
LP_WPW2   equ 40                ;   conv module out [D][D]
LP_W13B   equ 48                ;   ff2
LP_W2B    equ 56
LP_G13A   equ 64                ; their gradients, f32, in the same order
LP_G2A    equ 72
LP_GQKV   equ 80
LP_GO     equ 88
LP_GPW1   equ 96
LP_GPW2   equ 104
LP_G13B   equ 112
LP_G2B    equ 120
LP_NFA    equ 128               ; small params, f32 masters: the norms [D]
LP_NAT    equ 136
LP_NCV    equ 144
LP_NCI    equ 152               ; the one inside the conv module
LP_NFB    equ 160
LP_DW     equ 168               ; depthwise taps and bias [K+1][D]
LP_GNFA   equ 176               ; their gradients
LP_GNAT   equ 184
LP_GNCV   equ 192
LP_GNCI   equ 200
LP_GNFB   equ 208
LP_GDW    equ 216
LP_X      equ 224               ; activations saved for the backward. f32 [M][D]
LP_XNA    equ 232               ; T
LP_RSA    equ 240               ; rstd, f32 [M]
LP_HA     equ 248               ; T [M][2F]
LP_GA     equ 256               ; T [M][F]
LP_X1     equ 264
LP_XNT    equ 272
LP_RST    equ 280
LP_QKV    equ 288               ; T [M][3D]
LP_Y      equ 296               ; T [M][D]
LP_LSE    equ 304               ; f32 [B][H][T3]
LP_X2     equ 312
LP_XNC    equ 320
LP_RSC    equ 328
LP_HC     equ 336               ; T [M][2D]
LP_GC     equ 344               ; T [M][D]
LP_YC     equ 352               ; f32 [M][D], the depthwise conv's output
LP_ZC     equ 360               ; T, its norm
LP_RSZ    equ 368
LP_SC     equ 376               ; T, silu
LP_X3     equ 384
LP_XNB    equ 392
LP_RSB    equ 400
LP_HB     equ 408
LP_GB     equ 416
LP_SIZE   equ 424

; fe: the subsampling front end. mel [B][T][NMEL] -> conv1 (full 3x3 stride 2, relu)
; -> a1 -> depthwise 3x3 stride 2 -> b2 -> pointwise, relu -> a2 -> depthwise -> b3
; -> pointwise, relu -> a3, which as [B*T3][F3*C0] goes through WL into the encoder
FE_WL     equ 0                 ; matrices (matmul operands): [D][F3*C0]
FE_WP2    equ 8                 ;   [C0][C0]
FE_WP3    equ 16
FE_GWL    equ 24                ; their gradients
FE_GWP2   equ 32
FE_GWP3   equ 40
FE_W1     equ 48                ; conv taps and biases, f32 [10][C0]
FE_W2     equ 56
FE_W3     equ 64
FE_GW1    equ 72
FE_GW2    equ 80
FE_GW3    equ 88
FE_A1     equ 96                ; activations, T: [B][T1][F1][C0]
FE_B2     equ 104               ; [B][T2][F2][C0]
FE_A2     equ 112
FE_B3     equ 120               ; [B][T3][F3][C0]
FE_A3     equ 128
FE_D1     equ 136               ; backward scratch, T: the size of a1
FE_D2A    equ 144               ;   of a2 (two)
FE_D2B    equ 152
FE_D3A    equ 160               ;   of a3 (two)
FE_D3B    equ 168
FE_SIZE   equ 176

; one micro-batch, as model_fwd/bwd want it
MB_MEL    equ 0                 ; device f32 [B][T][NMEL], log-mel, normalized
MB_LAB    equ 8                 ; device u32 [B][UMAX], token ids
MB_TL     equ 16                ; device u32 [B], each utterance's encoder frames
MB_UL     equ 24                ; device u32 [B], its labels
MB_LOSS   equ 32                ; device f32 [B], where its losses (-ln p) go
MB_B      equ 40                ; u32 utterances
MB_T      equ 44                ; u32 mel frames (padded, all the same)
MB_UMAX   equ 48                ; u32 label row length in MB_LAB
MB_CS     equ 52                ; u32 attention chunk size, encoder frames (>= T3: full)
MB_LEFT   equ 56                ; u32 left chunks of context (~0: all)
MB_SCALE  equ 60                ; f32, 1 / labels in the whole step (the loss is a mean)
MB_SIZE   equ 64

; model_fwd modes
FW_LOGITS equ 0                 ; stop at the logits
FW_LOSS   equ 1                 ; ctc loss only
FW_GRAD   equ 2                 ; loss and dlogits, ready for model_bwd

; mm flags (both matmul kernels)
MM_TA     equ 1                 ; A stored [K][M]
MM_TB     equ 2                 ; B stored [K][N]
MM_BF     equ 4                 ; C is bf16
MM_F32    equ 8                 ; A and B are f32 (mm_ref only)

; model_step's settings, f64
OP_LR     equ 0
OP_B1     equ 8
OP_B2     equ 16
OP_EPS    equ 24
OP_WD     equ 32
OP_CLIP   equ 40
OP_SIZE   equ 48
