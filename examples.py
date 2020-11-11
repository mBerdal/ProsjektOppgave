import numpy as np
import matplotlib.pyplot as plt
from mission_space import MissionSpace
from helpers import get_visible_polygon, plot_cover_aux

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
  plot_cover_aux(V_i, ax, clr="#abcfff", alpha=1)
  V_bar_i = M.difference(V_i).difference(V_i.boundary)
  plot_cover_aux(V_bar_i, ax, clr="#ededed", alpha=1)
  M.plot(ax)
  ax.scatter(*s, color="blue", alpha=1, zorder=100)
  ax.annotate("$\mathbf{s}_{i}$", xy=(s[0]+0.1, s[1]+0.1))
  ax.annotate("$o_{0}$", xy=(2-0.1, 2))
  ax.annotate("$o_{1}$", xy=(-0.41, 0))
  ax.annotate("$V(\mathbf{s}_{i})$", xy=(-2, 2))
  ax.annotate("$\delta\Omega$", xy=(4-0.5, -3+0.1))
  ax.axis("equal")
  ax.axis("off")
  plt.savefig('../oppgave_tekst/figs/vis_set_example1.pdf', format='pdf')
  plt.show()