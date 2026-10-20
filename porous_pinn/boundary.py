import jax.numpy as jnp


def sample_boundaries_natural_convection(num_pts_per_side=128):
  """Sinh tọa độ và giá trị đích cho 4 cạnh của bài toán Natural Convection."""
  coords = jnp.linspace(0.0, 1.0, num_pts_per_side)

  # 1. Vách trái (x=0): u=0, v=0, T=1.0
  left_xy = jnp.column_stack([jnp.zeros_like(coords), coords])
  left_T = jnp.ones((num_pts_per_side, 1))

  # 2. Vách phải (x=1): u=0, v=0, T=0.0
  right_xy = jnp.column_stack([jnp.ones_like(coords), coords])
  right_T = jnp.zeros((num_pts_per_side, 1))

  # 3. Vách đáy (y=0): u=0, v=0, dT/dy=0
  bottom_xy = jnp.column_stack([coords, jnp.zeros_like(coords)])

  # 4. Vách đỉnh (y=1): u=0, v=0, dT/dy=0
  top_xy = jnp.column_stack([coords, jnp.ones_like(coords)])

  # Toàn bộ điểm vách cho vận tốc no-slip (4 cạnh)
  wall_xy = jnp.vstack([left_xy, right_xy, bottom_xy, top_xy])

  return {
      "wall_xy": wall_xy,
      "left_xy": left_xy,
      "left_T": left_T,
      "right_xy": right_xy,
      "right_T": right_T,
      "adiabatic_xy": jnp.vstack([bottom_xy, top_xy]),
  }
