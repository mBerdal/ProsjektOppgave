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

def plot_trilateration():
  S = np.array([
    [0, 1, 2],
    [0, 1, 0]
  ])

  p = np.array([0.3, 0.6])
  _, ax = plt.subplots()
  ax.scatter(*S, color="b")
  D = np.linalg.norm(p.reshape(2, 1) - S, axis=0)
  theta = np.linspace(0, 2*np.pi)
  for i in range(3):
    ax.plot(S[0, i] + D[i]*np.cos(theta), S[1, i] + D[i]*np.sin(theta), "k--")
    ax.plot(
      [S[0, i], S[0, i] + D[i]*np.cos(0)],
      [S[1, i], S[1, i] + D[i]*np.sin(0)],
      "b"
    )
    str_d = "$d_{" + str(i) + "}$"
    ax.annotate(str_d, xy = (S[0, i] + D[i]/2, S[1, i]+0.1))
    str_s = "$\mathbf{S}_{" + str(i) + "}$"
    ax.annotate(str_s, xy = (S[0, i], S[1, i]+0.05))
  ax.scatter(*p, color="#4eb53e", zorder=100)
  ax.annotate("$\mathbf{p}$", xy=(p[0]+0.1, p[1]))
  ax.set_facecolor("#ededed")
  ax.axis("equal")
  ax.axis("off")
  plt.savefig('report/figs/trilateration_example.pdf', format='pdf')
  plt.show()

if __name__ == "__main__":
  plot_trilateration()