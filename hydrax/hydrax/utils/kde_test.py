import jax
import jax.numpy as jnp
import numpy as np

from hydrax.utils.kde import gaussian_kde
# from jax.scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

dataset = jax.random.multivariate_normal(jax.random.PRNGKey(0),0*jnp.ones(1), jnp.array([[1]]), (512,))
# dataset = jnp.ones_like(jax.random.multivariate_normal(jax.random.PRNGKey(0),0*jnp.ones(1), jnp.array([[1]]), (512,)))
kde = gaussian_kde(dataset=dataset.T, bw=0.2)

pdf = kde.pdf(jnp.arange(-3,3,0.01,dtype=float).T)
plt.figure()
plt.plot(jnp.arange(-3,3,0.01,dtype=float), pdf)
plt.hist(dataset,bins=20,density=True)
plt.show()

# plt.hist2d(dataset[:,0],dataset[:,1], bins=30)


# x = dataset[:, 0]
# y = dataset[:, 1]

# # 2D histogram
# bins = 30
# counts, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)

# # Bin centers
# xpos = (xedges[:-1] + xedges[1:]) / 2
# ypos = (yedges[:-1] + yedges[1:]) / 2
# xposM, yposM = np.meshgrid(xpos, ypos, indexing='ij')

# # Flatten for bar3d
# xposF = xposM.ravel()
# yposF = yposM.ravel()
# zposF = np.zeros_like(xposF)

# # Bar sizes
# dx = np.diff(xedges).mean() * np.ones_like(zposF)
# dy = np.diff(yedges).mean() * np.ones_like(zposF)
# dz = counts.ravel()

# # Plot
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.bar3d(xposF, yposF, zposF, dx, dy, dz, shade=True)

# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Count")
# ax.set_aspect("auto")
# plt.show()
