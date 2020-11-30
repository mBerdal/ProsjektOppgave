from matplotlib.pyplot import box
from obstacle import *
from mission_space import MissionSpace
from helpers import get_covered_polygon, get_visible_polygon, plot_cover, plot_visible_polygon
import numpy as np
import matplotlib.pyplot as plt
bg_clr = "#ededed"
plt.rcParams['figure.facecolor'] = bg_clr
from global_opt import GlobalOpt
from distr_opt import DistrOpt

def plot_distr(X, M, com_radius, ax):
  cover = get_covered_polygon(X, com_radius, mission_space=M)
  M.plot(ax)
  plot_cover(cover, ax)
  ax.set_title("Final configuration")
  ax.scatter(X[0, :], X[1, :], zorder=100, color="orange")
  for i in np.arange(X.shape[1]):
    plot_visible_polygon(get_visible_polygon(X[:, i], com_radius, M), ax)


def all_spawn(M, N_dots, com_radius, box_bounds, min_dist):
  config_fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
  config_fig.set_size_inches(12, 6)
  xmin, ymin, xmax, ymax = M.bounds
  axs[0].set_xlim(xmin-1, xmax+1)
  axs[0].set_ylim(ymin-1, ymax+1)

  X = np.random.uniform(-box_bounds, -box_bounds + 1, 2*N_dots).reshape(2, N_dots)
  plot_distr(X, M, com_radius, axs[0])

  G = DistrOpt(X, com_radius, M, min_dist, k_1=1)
  G.optimize()
  X_star = G.X_star

  area_traj_fig, area_ax = plt.subplots()
  area_traj_fig.set_size_inches(7, 6)
  area_ax.plot(G.area_traj[1, :], G.area_traj[0, :]/M.area)
  area_ax.set_ylabel("Covered area [% of area of $\mathcal{F}$]")
  area_ax.set_yticklabels([f"{100*v:.1f}%" for v in area_ax.get_yticks()])
  area_ax.set_xlabel("Iteration")

  plot_distr(X_star, M, com_radius, axs[1])

  print("---X_star---")
  print(X_star)

  config_fig.savefig("report/figs/6_by_6_bottom_left_wall_obs_20_agnts_distr.pdf", format="pdf", dpi=config_fig.dpi)
  area_traj_fig.savefig("report/figs/6_by_6_bottom_left_wall_obs_20_agnts_area_traj.pdf", format="pdf", dpi=area_traj_fig.dpi)

  plt.show()


def single_spawn(M, N_dots, com_radius, box_bounds, min_dist):
  X = np.random.uniform(-box_bounds, -box_bounds + 0.2, 4).reshape(2, 2)
  for a in range(N_dots):
    print("starting", a)
    X = np.hstack((X, np.random.uniform(-box_bounds+0.2, -box_bounds + 0.4, 2).reshape(2, 1)))
    G = DistrOpt(X, com_radius, M, min_dist)
    X = G.one_at_the_time_optimize()

  _, ax = plt.subplots()
  plot_distr(X, M, com_radius, ax)
  plt.show()



if __name__=="__main__":
  N_dots = 20
  com_radius = 3
  box_bounds = 5
  min_dist = 0.2

  

  obstacles = [Obstacle.get_bottom_left_wall(box_bounds)]

  M = MissionSpace(
    np.array([
      [-box_bounds, -box_bounds],
      [box_bounds, -box_bounds],
      [box_bounds, box_bounds],
      [-box_bounds, box_bounds]
    ]), obstacles)

  all_spawn(M, N_dots, com_radius, box_bounds, min_dist)


  