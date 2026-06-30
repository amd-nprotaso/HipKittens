import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from kernel import build_gqa_attn

torch.manual_seed(0)
B, D, H, H_KV, N = 1, 128, 8, 2, 256
dtype = torch.bfloat16

q = torch.randn(B, N, H, D, dtype=dtype, device="cuda")
k = torch.randn(B, N, H_KV, D, dtype=dtype, device="cuda")
v = torch.randn(B, N, H_KV, D, dtype=dtype, device="cuda")
out = torch.zeros(B, N, H, D, dtype=dtype, device="cuda")
lse = torch.zeros(B, H, 1, N, dtype=torch.float32, device="cuda")

launch = build_gqa_attn(ATTN_B=B, ATTN_H=H, ATTN_H_KV=H_KV, ATTN_N=N, ATTN_D=D)
stream = torch.cuda.current_stream()


def args(q, k, v, out, lse):
    return (
        q.reshape(-1), k.reshape(-1), v.reshape(-1), out.reshape(-1), lse.reshape(-1),
        H * D, H_KV * D, H_KV * D, H * D, N, fx.Stream(stream),
    )


compiled = flyc.compile(launch, *args(q, k, v, out, lse))
print("compiled OK")
compiled(*args(q, k, v, out, lse))
torch.cuda.synchronize()
print("ran OK")

G = H // H_KV
qh = q.transpose(1, 2).float()
kh = k.transpose(1, 2).repeat_interleave(G, dim=1).float()
vh = v.transpose(1, 2).repeat_interleave(G, dim=1).float()
ref = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh).transpose(1, 2).to(dtype)

diff = (out.float() - ref.float()).abs()
cos = torch.nn.functional.cosine_similarity(out.flatten().float(), ref.flatten().float(), dim=0).item()
print(f"max_abs={diff.max().item():.5f} mean_abs={diff.mean().item():.5f} cos={cos:.6f}")
print("TK :", out[0, 0, :8, 0])
print("ref:", ref[0, 0, :8, 0])

scale = 1.0 / (D ** 0.5)
scores = (qh @ kh.transpose(-1, -2)) * scale
lse_ref = torch.logsumexp(scores, dim=-1)
lse_fly = lse[:, :, 0, :]
ldiff = (lse_fly - lse_ref).abs()
print(f"LSE max_abs={ldiff.max().item():.5f} mean={ldiff.mean().item():.6f}")
