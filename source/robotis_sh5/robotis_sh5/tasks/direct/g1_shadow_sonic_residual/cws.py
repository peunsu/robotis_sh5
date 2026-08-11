"""접촉을 "물체를 어떻게 움직일 수 있는가"로 바꾸는 계산 (CHORD, arXiv 2607.00033).

용어를 먼저 풀어 씁니다.

- 접촉 하나가 물체에 주는 효과는 두 가지입니다: 미는 힘(3방향)과 그 힘이 물체 중심에 대해
  만드는 회전 효과(3축). 합쳐서 숫자 6개이고, 논문은 이 묶음을 렌치(wrench)라고 부릅니다.
- 손가락은 표면에 수직으로만 미는 게 아니라 마찰 덕분에 비스듬히도 밉니다. 밀 수 있는 방향의
  범위가 원뿔이 됩니다(마찰 원뿔). 반각은 atan(마찰계수)입니다.
- 손 전체의 능력은 미리 정해둔 방향 512개 각각에 대해 "이 방향으로 최대 얼마나 밀 수 있나"를
  기록한 목록으로 요약합니다. 논문 용어로는 지지함수(support function)입니다.

기본 경로(support)는 논문 방식 그대로 원뿔을 대표 방향으로 근사합니다. 재현이 목적이기 때문입니다.
support_exact는 원뿔 위 최대값의 닫힌 형태로, 근사가 맞게 구현됐는지 대조하는 용도로 둡니다
(대표 방향을 128개로 늘리면 0.01% 이내로 일치).

논문과 다른 점이 하나 있습니다.

1. 회전 성분을 물체 크기로 나눕니다. 6개 숫자 중 앞 3개(밀기)는 크기가 1인데 뒤 3개(회전)는
   접촉점이 물체 중심에서 떨어진 거리에 비례한 미터 값이라, 그냥 두면 칼(5.6 cm)에서는 회전
   능력이 밀기의 1/18로 묻히고 팬(15.0 cm)에서는 1/7이 되어 같은 조절값이 두 클립에서 다른
   뜻을 갖습니다. 논문에는 이 처리가 없습니다.
"""

import torch


def make_basis(n_dir: int = 512, seed: int = 0, device="cpu") -> torch.Tensor:
    """비교에 쓸 방향 n_dir개. (6, n_dir).

    사람 쪽과 로봇 쪽이 반드시 같은 방향 집합을 써야 하므로 시드를 고정합니다.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    b = torch.randn(6, n_dir, generator=g)
    return (b / b.norm(dim=0, keepdim=True)).to(device)


def support_exact(basis, points, normals, active, mu: float, length_scale: float) -> torch.Tensor:
    """방향마다 "최대 얼마나 밀 수 있나". -> (..., n_dir).

    basis (6, n_dir), points/normals (..., L, 3), active (..., L).
    points와 normals는 물체 기준 좌표여야 합니다(물체가 움직여도 값이 안 변하도록).
    normals는 접촉면이 물체를 미는 방향입니다.

    유도. 원뿔 안의 단위 힘 f가 방향 b=(b_f, b_m)에 기여하는 값은

        b·w(f) = b_f·f + b_m·(p×f)/L = f·g,    g = b_f + (b_m×p)/L

    (스칼라 삼중곱 b_m·(p×f) = f·(b_m×p)). 즉 f·g를 원뿔 안에서 최대화하는 문제이고, g가 원뿔
    안이면 |g|, 밖이면 |g|cos(θ-α)입니다(θ = g와 법선 사이 각, α = 원뿔 반각). 접촉이 없거나
    최대값이 음수인 경우는 "안 민다"는 선택지가 있으므로 0입니다.

    g를 직접 만들면 (..., L, n_dir, 3) 크기의 중간 텐서가 생겨 환경 수천 개에서 메모리를
    잡아먹으므로, 필요한 스칼라량만 삼중곱으로 풀어 (..., L, n_dir) 행렬곱 네 번으로 구합니다.
    """
    bf, bm = basis[:3].transpose(0, 1), basis[3:].transpose(0, 1)      # (n_dir,3)
    Ls = float(length_scale)
    n = normals / normals.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    pxn = torch.cross(points, n, dim=-1)                               # (...,L,3)
    bfxbm = torch.cross(bf, bm, dim=-1)                                # (n_dir,3)
    # g·n = b_f·n + b_m·(p×n)/L        ((b_m×p)·n = b_m·(p×n))
    gn = n @ bf.transpose(0, 1) + (pxn @ bm.transpose(0, 1)) / Ls
    # |g|² = |b_f|² + 2 p·(b_f×b_m)/L + (|b_m|²|p|² - (b_m·p)²)/L²
    bm_p = points @ bm.transpose(0, 1)                                 # (...,L,n_dir)
    g2 = (bf.pow(2).sum(-1)                                            # (n_dir,)
          + 2.0 * (points @ bfxbm.transpose(0, 1)) / Ls
          + (bm.pow(2).sum(-1) * points.pow(2).sum(-1, keepdim=True) - bm_p.pow(2)) / (Ls * Ls))
    gnorm = g2.clamp(min=0.0).sqrt()

    ca = 1.0 / (1.0 + mu * mu) ** 0.5                                  # cos(원뿔 반각)
    sa = mu * ca                                                       # sin(원뿔 반각)
    cos_t = (gn / gnorm.clamp(min=1e-9)).clamp(-1.0, 1.0)              # g와 법선 사이 각의 cos
    sin_t = (1.0 - cos_t * cos_t).clamp(min=0.0).sqrt()
    # g가 원뿔 안이면 |g|, 밖이면 |g|cos(θ-α)
    val = gnorm * torch.where(cos_t >= ca, torch.ones_like(cos_t), cos_t * ca + sin_t * sa)
    val = val.clamp(min=0.0) * active.unsqueeze(-1).to(val.dtype)
    return val.max(dim=-2).values                                      # 링크에 대한 최대 -> (...,n_dir)


def cws_reward(sigma_h, sigma_r, beta: float, v: float) -> torch.Tensor:
    """사람 목록과 로봇 목록을 비교해 0~1 점수로. 둘 다 (..., n_dir).

    로봇이 사람의 (1-beta)배보다 못 밀면 부족으로, (1+beta)배를 넘겨 밀면 과잉으로 감점합니다.
    그 사이 여유 범위 안에서는 두 항이 모두 0이라 만점입니다.
    """
    lo = ((1.0 - beta) * sigma_h - sigma_r).clamp(min=0.0)
    hi = (sigma_r - (1.0 + beta) * sigma_h).clamp(min=0.0)
    return torch.exp(-(lo.pow(2).sum(-1) + hi.pow(2).sum(-1)) / v)


# ---------------------------------------------------------------------------------------------
# 논문 방식(원뿔을 대표 방향으로 근사). 재현이 목적이므로 이쪽이 기본 경로입니다.

def friction_cone(normal, mu: float, n_edge: int = 4):
    """마찰 원뿔의 대표 방향들. normal (...,3) -> (...,n_edge,3), 각각 크기 1."""
    n = normal / normal.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    ref = torch.zeros_like(n)
    ref[..., 2] = 1.0
    alt = torch.zeros_like(n)
    alt[..., 0] = 1.0
    ref = torch.where((n[..., 2:3].abs() > 0.9).expand_as(ref), alt, ref)
    t1 = torch.cross(n, ref, dim=-1)
    t1 = t1 / t1.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    t2 = torch.cross(n, t1, dim=-1)
    phi = torch.arange(n_edge, device=n.device, dtype=n.dtype) * (2 * torch.pi / n_edge)
    tang = torch.cos(phi)[:, None] * t1.unsqueeze(-2) + torch.sin(phi)[:, None] * t2.unsqueeze(-2)
    e = n.unsqueeze(-2) + mu * tang
    return e / e.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def support(basis, points, normals, active, mu: float, length_scale: float, n_edge: int = 16,
            link_chunk: int = 0):
    """논문 방식: 원뿔을 대표 방향 n_edge개로 근사하고 그중 최대. -> (..., n_dir).

    n_edge는 논문에 값이 없어 우리가 정해야 합니다. 대표 방향을 뽑으려면 고정 기준축이 필요한데
    그래서 물체를 회전시키면 값이 달라집니다 — 물리가 아니라 근사 때문에 생기는 오차이고, 실측
    상대오차(중앙/최대)는 4개 5.1%/42%, 8개 1.1%/10.2%, 16개 0.3%/2.5%, 32개 0.08%/0.6%입니다.
    우리 물체는 클립 내내 자세가 바뀌므로 같은 파지가 프레임마다 다른 점수를 받게 됩니다. 얻는
    것이 없는 오차라 감당 가능한 선에서 크게 잡습니다. 중간 배열이 (환경 x n_dir x 링크*n_edge)로
    n_edge에 비례해 커지는 것이 비용입니다.
    """
    L = points.shape[-2]
    chunk = int(link_chunk) if link_chunk else L
    Bt = basis.transpose(0, 1)                                         # (n_dir,6)
    out = None
    # 링크를 나눠 처리하고 최대만 누적합니다. 한 번에 하면 중간 배열이
    # (환경 x n_dir x 링크*n_edge)라, 환경 2048/방향 512/링크 32/대표방향 16에서 2 GB가 넘습니다.
    # 최대는 결합법칙이 성립하므로 나눠서 구해도 결과가 같습니다.
    for i in range(0, L, chunk):
        pc, nc, ac = points[..., i:i + chunk, :], normals[..., i:i + chunk, :], active[..., i:i + chunk]
        e = friction_cone(nc, mu, n_edge)
        pe = pc.unsqueeze(-2).expand_as(e)
        w = torch.cat([e, torch.cross(pe, e, dim=-1) / length_scale], dim=-1)
        w = w * ac.unsqueeze(-1).unsqueeze(-1).to(w.dtype)
        w = w.flatten(-3, -2).transpose(-1, -2)                        # (...,6,chunk*n_edge)
        v = (Bt @ w).max(dim=-1).values                                # (...,n_dir)
        out = v if out is None else torch.maximum(out, v)
    return out.clamp(min=0.0)
