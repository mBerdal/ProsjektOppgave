import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from shapely.geometry.point import Point
from mission_space import MissionSpace
from distr_opt import DistrOpt
from helpers import get_visible_polygon, plot_cover, plot_visible_polygon

bg_clr = "#ededed"
plt.rcParams['figure.facecolor'] = bg_clr

def plot_vis_set_example():
  s = np.array([-1, 0.5])
  a = 1
  b = np.sqrt(3)*a
  obs1 = np.array([
    [0, -2*a],
    [b, -a],
    [b, a],
    [0, 2*a],
    [-b, a],
    [-b, -a]
  ]) + 2*np.ones((6, 2))
  obs2 = np.array([
    [0.5, 0],
    [1, 0],
    [1, 2],
    [0.5, 2]
  ])- np.ones((4, 2))
  M = MissionSpace(np.array([
    [-3, -3],
    [4, -3],
    [4, 4],
    [-3, 4]
  ]), [obs1, obs2])
  _, ax = plt.subplots()
  V_i = get_visible_polygon(s, 3.4, M)
  M.plot(ax)
  plot_cover(V_i, ax, clr="#abcfff", alpha=1)
  ax.scatter(*s, color="blue", alpha=1, zorder=100)
  ax.annotate("$\mathbf{x}_{a}$", xy=(s[0]+0.1, s[1]+0.1))
  ax.annotate("$o_{0}$", xy=(2-0.1, 2))
  ax.annotate("$o_{1}$", xy=(-0.41, 0))
  ax.annotate("$V(\mathbf{x}_{a})$", xy=(-2, 2))
  ax.annotate("$\delta\Omega$", xy=(4-0.5, -3+0.1))
  ax.axis("equal")
  ax.axis("off")
  plt.savefig('report/figs/vis_set_example.pdf', format='pdf')
  plt.show()

def plot_trilateration():
  X = np.array([
    [0, 1, 2],
    [0, 1, 0]
  ])

  y = np.array([0.3, 0.6])
  _, ax = plt.subplots()
  ax.scatter(*X, color="b")
  D = np.linalg.norm(y.reshape(2, 1) - X, axis=0)
  theta = np.linspace(0, 2*np.pi, 10000)
  for i in range(3):
    ax.plot(X[0, i] + D[i]*np.cos(theta), X[1, i] + D[i]*np.sin(theta), "k--")
    ax.plot(
      [X[0, i], X[0, i] + D[i]*np.cos(0)],
      [X[1, i], X[1, i] + D[i]*np.sin(0)],
      "b"
    )
    str_d = "$d_{" + str(i) + "}$"
    ax.annotate(str_d, xy = (X[0, i] + D[i]/2, X[1, i]+0.1))
    str_x = "$\mathbf{x}_{" + str(i) + "}$"
    ax.annotate(str_x, xy = (X[0, i], X[1, i]+0.05))
  ax.scatter(*y, color="#4eb53e", zorder=100)
  ax.annotate("$\mathbf{y}$", xy=(y[0]+0.1, y[1]))
  ax.axis("equal")
  ax.axis("off")
  plt.savefig('report/figs/trilateration_example.pdf', format='pdf')
  plt.show()

def plot_intersections():
  M = MissionSpace(np.array([
    [-1.1, -1.1],
    [1.1, -1.1],
    [1.1, 1.1],
    [-1.1, 1.1]
  ]))
  fig, ax = plt.subplots()
  fig.set_size_inches(6, 6)
  M.plot(ax)
  rad = 0.5
  x_a = (-0.5, 0)
  c_a = Point(*x_a).buffer(rad)
  ax.scatter(*x_a, color="orange", zorder=100, label="$\mathbf{x}_{a}$")
  plot_visible_polygon(c_a, ax)
  x_1 = (0, -0.25)
  c_1 = Point(*x_1).buffer(rad)
  ax.scatter(*x_1, color="blue", zorder=100)
  plot_visible_polygon(c_1, ax)
  x_2 = (0, 0.25)
  c_2 = Point(*x_2).buffer(rad)
  ax.scatter(*x_2, color="blue", zorder=100)
  plot_visible_polygon(c_2, ax)
  x_3 = (0.25, 0.2)
  c_3 = Point(*x_3).buffer(rad)
  ax.scatter(*x_3, color="blue", zorder=100, label="$\mathbf{x}_{j},\;j\in\mathcal{N}/\{a\}$")
  plot_visible_polygon(c_3, ax)
  int_a_1 = c_a.intersection(c_1).intersection(c_2).difference(c_3)
  ax.fill(*int_a_1.exterior.xy, color="blue", alpha=0.3, label="$\{\mathbf{y}:\:\Phi_{\mathcal{N}/\{a\}}^{2}(\mathbf{y})\^{p}_{a}(\mathbf{y}) > 0\}$")
  int_a_3 = c_a.intersection(c_2).intersection(c_3).difference(c_1)
  ax.fill(*int_a_3.exterior.xy, color="blue", alpha=0.3)
  int_neighs = c_1.intersection(c_2).intersection(c_3)
  ax.fill(*int_neighs.exterior.xy, color="green", alpha=0.3, label="$\{\mathbf{y}:\:\Phi_{\mathcal{N}/\{a\}}^{3^{+}}(\mathbf{y}) > 0\}$")

  circs = [c_a, c_1, c_2, c_3]
  from itertools import combinations
  u = None
  for x, y, z in combinations(circs, 3):
    if u is None:
      u = x.intersection(y).intersection(z)
    else:
      u = u.union(x.intersection(y).intersection(z))
  ax.plot(*u.exterior.xy, color="black", label="$\delta\{\mathbf{y}:\:\Phi_{\mathcal{N}}^{3^{+}}(\mathbf{y}) > 0\}$")

  ax.axis("equal")
  ax.axis("off")
  ax.legend()
  plt.savefig('report/figs/local_objective_example.pdf', format='pdf')

  plt.show()

def plot_close_dist_repell():
  k_1 = 1
  k_2s = np.linspace(0.5, 2.7, 3)
  x_a = np.zeros((1, 1))
  cdr = lambda x, k_1, k_2: k_1*np.exp(-k_2*np.linalg.norm(x_a - x, axis=0)).reshape(x.shape)
  x = linspace(-5, 5, 10000).reshape(1, 10000)
  _, ax = plt.subplots()
  for k_2 in k_2s:
    ax.plot(x[0, :], cdr(x, k_1, k_2)[0, :], color="#155c8f")
    ax.annotate(f"$k_{2} = {k_2}$", xy = (x[0, 6000], cdr(x, k_1, k_2)[0, 6000]))
  ax.plot(x[0, :], k_1*np.ones((x.shape[1], )), color="black", alpha=0.5, linestyle="dashed")
  ax.set_xlabel("$||\mathbf{x}_{a}-\mathbf{x}_{j}||$")
  ax.set_yticks([0, k_1])
  ax.set_yticklabels(["$0$", "$k_{1}$"])
  ax.set_xticks([-4, -2, 0, 2, 4])
  ax.set_xticklabels(["$-4$", "$-2$", "$0$", "$2$", "$4$"])
  plt.savefig('report/figs/close_dist_repell_example.pdf', format='pdf')
  plt.show()

def plot_objective():
  N_B_a = 3
  X_B_a = np.random.uniform(-10, 10, 2*N_B_a).reshape(2, N_B_a)
  M = MissionSpace(np.array([
    [-10, -10],
    [10, -10],
    [10, 10],
    [-10, 10],
  ]))
  print(X_B_a)
  V_B_a = [get_visible_polygon(X_B_a[:, j], 3, M) for j in np.arange(X_B_a.shape[1])]
  fig = plt.figure()
  ax = fig.add_subplot(111, projection="3d")
  ys = np.linspace(-10, 10, 50)
  xs = np.linspace(-10, 10, 50)
  xx, yy = np.meshgrid(xs, ys)
  zz = np.zeros(xx.shape)
  for x in np.arange(xs.shape[0]):
    for y in np.arange(ys.shape[0]):
      zz[y, x] = DistrOpt.area_covered_by_two_neighbours(np.array([xs[x], ys[y]]), V_B_a, 3, M) - DistrOpt.close_dist_repell(np.array([xs[x], ys[y]]), X_B_a)
  
  for i in np.arange(X_B_a.shape[1]):
    print(X_B_a[0, i], X_B_a[1, i])
    ax.scatter(X_B_a[0, i], X_B_a[1, i], -1, color="orange")
  ax.set_xlabel("$x_{a}$")
  ax.set_ylabel("$y_{a}$")
  ax.plot_surface(xx, yy, zz)
  plt.show()
  pass


if __name__ == "__main__":
  plot_objective()
  exit(0)
  plot_close_dist_repell()
  plot_intersections()
  plot_trilateration()
  plot_vis_set_example()