from obstacle import *
from mission_space import MissionSpace
from helpers import get_covered_polygon, plot_cover_aux, get_visible_polygon
import numpy as np
import matplotlib.pyplot as plt
from global_opt import GlobalOpt
from distr_opt import DistrOpt

if __name__=="__main__":
  N_dots = 3
  com_radius = 9
  box_bounds = 3
  min_dist = 0.2

  obstacles = [Obstacle.get_centered_box(box_bounds, 3)]

  M = MissionSpace(
    np.array([
      [-box_bounds, -box_bounds],
      [box_bounds, -box_bounds],
      [box_bounds, box_bounds],
      [-box_bounds, box_bounds]
    ]), [])

  _, axs = plt.subplots(2, sharex=True, sharey=True)
  xmin, ymin, xmax, ymax = M.bounds
  axs[0].set_xlim(xmin-1, xmax+1)
  axs[0].set_ylim(ymin-1, ymax+1)

  S = np.random.uniform(-box_bounds, -box_bounds + 1, 2*N_dots).reshape(2, N_dots)
  initial_cover = get_covered_polygon(S, com_radius, mission_space=M)
  M.plot(axs[0])
  plot_cover_aux(initial_cover, axs[0])
  axs[0].set_title("Initial")
  axs[0].scatter(S[0, :], S[1, :])
  for i in np.arange(N_dots):
    plot_cover_aux(get_visible_polygon(S[:, i], com_radius, M), axs[0], clr="green", alpha=0.1)

  G = GlobalOpt(S, com_radius, M, min_dist)
  G.optimize()
  S_star = G.S_star

  final_cover = get_covered_polygon(S_star, com_radius, mission_space=M)
  M.plot(axs[1])
  plot_cover_aux(final_cover, axs[1])
  axs[1].set_title("Final")
  axs[1].scatter(S_star[0, :], S_star[1, :])
  for i in np.arange(N_dots):
    plot_cover_aux(get_visible_polygon(S_star[:, i], com_radius, M), axs[1], clr="green", alpha=0.1)

  print("---S_star---")
  print(S_star)
  print("initial cover: ",initial_cover.area)
  print("final cover: ",final_cover.area)

  plt.show()