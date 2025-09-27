import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm  # centers colormap at 0

# --- example data (replace with your x0, x1, Ls) ---
n = 60
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
x0, x1 = np.meshgrid(x, y)
Ls = np.exp(-(x0**2 + x1**2)) * (np.cos(x0**2 + x1**2))  # example scalar field

# Plot
fig, ax = plt.subplots(figsize=(6, 6))

# make cmap symmetric around zero
max_abs = np.max(np.abs(Ls))
norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

pcm = ax.pcolormesh(x0, x1, Ls, cmap="RdBu_r", shading="auto", norm=norm)

ax.set_aspect("equal")
ax.set_xlabel("x0")
ax.set_ylabel("x1")
fig.colorbar(pcm, ax=ax, label="Ls")

plt.tight_layout()
plt.show()
