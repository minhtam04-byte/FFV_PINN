"""Collocation-point batches for the 2D porous-convection cases.

Moved verbatim from the case notebooks; also used by scripts/train.py.
"""
import itertools

import jax
import jax.numpy as jnp

from jaxpi.samplers import MeshSampler, UniformSampler


def create_batches(config):
    H, L = float(config.physics.H), float(config.physics.L)
    scheme = getattr(config.diff, "scheme", "ad").lower().strip()

    if scheme in ["ffv", "can_ffv", "ffv_continuous", "ffv_compact"]:
        n_x = config.diff.n_x
        n_y = config.diff.n_y

        dx = L / n_x
        dy = H / n_y

        assert jnp.isclose(dx, dy, atol=1e-6), (
            f"Sơ đồ {scheme.upper()} yêu cầu lưới vuông dx == dy, hiện tại dx={dx} != dy={dy}"
        )

        config.diff.h_stencil = dx

        # ffv, ffv_continuous và can_ffv dùng tâm ô lân cận E, W, N, S -> reach = h;
        # ffv_compact chỉ dùng các mặt e, w, n, s -> reach = h/2, phủ cả các ô sát vách
        if scheme == "ffv_compact":
            stencil_reach = 0.5
            num_exclude = 0
        else:
            stencil_reach = 1.0
            num_exclude = 1

        use_continuous_mesh = getattr(config.diff, "use_continuous_mesh", False)

        if use_continuous_mesh:
            safe_x = stencil_reach * dx + 1e-4
            safe_y = stencil_reach * dy + 1e-4
            domain_safe = jnp.array([[safe_x, L - safe_x], [safe_y, H - safe_y]])
            sampler = UniformSampler(domain_safe, config.training.batch_size)
            if config.training.random_sampling:
                return sampler
            return itertools.repeat(next(iter(sampler)))

        else:
            num_pts_x = n_x - 2 * num_exclude
            num_pts_y = n_y - 2 * num_exclude

            assert num_pts_x > 0 and num_pts_y > 0, (
                f"Số lượng ô {n_x}x{n_y} quá ít so với stencil reach {stencil_reach}"
            )

            x_start = (num_exclude + 0.5) * dx
            x_end = L - (num_exclude + 0.5) * dx
            y_start = (num_exclude + 0.5) * dy
            y_end = H - (num_exclude + 0.5) * dy

            x_inner = jnp.linspace(x_start, x_end, num_pts_x)
            y_inner = jnp.linspace(y_start, y_end, num_pts_y)

            X, Y = jnp.meshgrid(x_inner, y_inner, indexing="ij")
            mesh_points = jnp.column_stack([X.ravel(), Y.ravel()])

            total_mesh_pts = mesh_points.shape[0]
            num_devices = jax.device_count()

            # QUAN TRỌNG: Nếu tổng số ô lưới <= batch_size, trả về TOÀN BỘ 100% điểm lưới
            # (cắt cho chia hết số GPU nếu cần) để tránh lỗi lấy mẫu trùng lặp bỏ sót 36.8% số ô của MeshSampler!
            if total_mesh_pts <= config.training.batch_size:
                usable_pts = (total_mesh_pts // num_devices) * num_devices
                return itertools.repeat(mesh_points[:usable_pts])
            else:
                sampler = MeshSampler(
                    mesh=mesh_points, batch_size=config.training.batch_size
                )
                if config.training.random_sampling:
                    return sampler
                return itertools.repeat(next(iter(sampler)))

    elif scheme == "ad":
        domain = jnp.array([[0.0, L], [0.0, H]])
        sampler = UniformSampler(domain, config.training.batch_size)
        if config.training.random_sampling:
            return sampler
        return itertools.repeat(next(iter(sampler)))
    else:
        raise ValueError(f"Sơ đồ diff.scheme='{scheme}' không được hỗ trợ.")
