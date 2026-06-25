import torch, sys
import flydsl.compiler as flyc
import flydsl.expr as fx
from kernel import build_gqa_attn

torch.manual_seed(0)
B, D, H, H_KV, N = 16, 128, 64, 8, 2048
dtype = torch.bfloat16

q = torch.randn(B, N, H, D, dtype=dtype, device='cuda')
k = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda')
v = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda')
out = torch.zeros(B, N, H, D, dtype=dtype, device='cuda')
lse = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda')

launch = build_gqa_attn(ATTN_B=B, ATTN_H=H, ATTN_H_KV=H_KV, ATTN_N=N, ATTN_D=D)
stream = torch.cuda.current_stream()

def args(q, k, v, out, lse):
    return (q.reshape(-1), k.reshape(-1), v.reshape(-1), out.reshape(-1), lse.reshape(-1),
            H * D, H_KV * D, H_KV * D, H * D, N, fx.Stream(stream))

compiled = flyc.compile(launch, *args(q, k, v, out, lse))
print("compiled OK")
compiled(*args(q, k, v, out, lse))
torch.cuda.synchronize()
print("ran OK")

# reference (GQA: head h maps to kv-head h // GROUP_SIZE)
G = H // H_KV
qh = q.transpose(1, 2).float()                       # [B,H,N,D]
kh = k.transpose(1, 2).repeat_interleave(G, dim=1).float()
vh = v.transpose(1, 2).repeat_interleave(G, dim=1).float()
ref = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh).transpose(1, 2).to(dtype)

diff = (out.float() - ref.float()).abs()
cos = torch.nn.functional.cosine_similarity(out.flatten().float(), ref.flatten().float(), dim=0).item()
print(f"max_abs={diff.max().item():.5f} mean_abs={diff.mean().item():.5f} cos={cos:.6f}")
print("TK :", out[0,0,:8,0])
print("ref:", ref[0,0,:8,0])

# LSE reference: logsumexp over scaled scores, natural log
scale = 1.0 / (D ** 0.5)
scores = (qh @ kh.transpose(-1, -2)) * scale          # [B,H,N,N]
lse_ref = torch.logsumexp(scores, dim=-1)             # [B,H,N]
lse_fly = lse[:, :, 0, :]                              # [B,H,N]
ldiff = (lse_fly - lse_ref).abs()
print(f"LSE max_abs={ldiff.max().item():.5f} mean={ldiff.mean().item():.6f}")
print("LSE TK :", lse_fly[0,0,:6])
print("LSE ref:", lse_ref[0,0,:6])

# timing
import time
for _ in range(20):
    compiled(*args(q, k, v, out, lse))
torch.cuda.synchronize()
it = 100
t0 = time.time()
for _ in range(it):
    compiled(*args(q, k, v, out, lse))
torch.cuda.synchronize()
ms = (time.time()-t0)/it*1e3
flop = 4 * B * N*N * H * D
print(f"FlyDSL: {ms:.4f} ms  {flop/1e12/(ms/1e3):.1f} TFLOPS")
