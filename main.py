from obstacle import *
from mission_space import MissionSpace
from helpers import get_covered_polygon, get_visible_polygon, plot_cover, plot_visible_polygon
import numpy as np
import matplotlib.pyplot as plt
bg_clr = "#ededed"
plt.rcParams['figure.facecolor'] = bg_clr
from global_opt import GlobalOpt
from distr_opt import DistrOpt

if __name__=="__main__":
  N_dots = 6
  com_radius = 2
  box_bounds = 3
  min_dist = 0.2

  obstacles = [Obstacle.get_centered_box(box_bounds, 50) + 2]

  M = MissionSpace(
    np.array([
      [-box_bounds, -box_bounds],
      [box_bounds, -box_bounds],
      [box_bounds, box_bounds],
      [-box_bounds, box_bounds]
    ]), obstacles)


  _, axs = plt.subplots(1, 2, sharex=True, sharey=True)
  xmin, ymin, xmax, ymax = M.bounds
  axs[0].set_xlim(xmin-1, xmax+1)
  axs[0].set_ylim(ymin-1, ymax+1)

  X = np.random.uniform(-box_bounds, -box_bounds + 1, 2*N_dots).reshape(2, N_dots)
  initial_cover = get_covered_polygon(X, com_radius, mission_space=M)
  M.plot(axs[0])
  plot_cover(initial_cover, axs[0])
  axs[0].set_title("Initial configuration")
  axs[0].scatter(X[0, :], X[1, :])
  for i in np.arange(N_dots):
    plot_visible_polygon(get_visible_polygon(X[:, i].reshape(2, 1), com_radius, M), axs[0])

  G = DistrOpt(X, com_radius, M, min_dist, 0)
  G.optimize()
  X_star = G.X_star

  _, area_ax = plt.subplots()
  area_ax.plot(G.area_traj[1, :], G.area_traj[0, :]/M.area)
  area_ax.set_ylabel("Covered area [%]")
  area_ax.set_xlabel("Iteration")

  final_cover = get_covered_polygon(X_star, com_radius, mission_space=M)
  M.plot(axs[1])
  plot_cover(final_cover, axs[1])
  axs[1].set_title("Final configuration")
  axs[1].scatter(X_star[0, :], X_star[1, :])
  for i in np.arange(N_dots):
    plot_visible_polygon(get_visible_polygon(X_star[:, i], com_radius, M), axs[1])

  print("---X_star---")
  print(X_star)
  print("initial cover: ",initial_cover.area)
  print("final cover: ",final_cover.area)

  plt.show()