from matplotlib.pyplot import box
from obstacle import *
from mission_space import MissionSpace
from helpers import get_covered_polygon, get_visible_polygon, plot_cover, plot_visible_polygon, plot_distr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
bg_clr = "#ededed"
plt.rcParams['figure.facecolor'] = bg_clr
from global_opt import GlobalOpt
from distr_opt import DistrOpt

def all_spawn(M, X, com_radius, min_dist, d_min, k_1, k_2, fignames = "trash", plot_steps = False):
  config_fig, axs = plt.subplots(1, 2)
  config_fig.set_size_inches(12, 6)
  xmin, ymin, xmax, ymax = M.bounds
  axs[0].set_xlim(xmin-1, xmax+1)
  axs[0].set_ylim(ymin-1, ymax+1)
  axs[0].set_title("Initial configuration")
  axs[0].axis("equal")
  axs[1].axis("equal")

  plot_distr(X, M, com_radius, axs[0])

  G = DistrOpt(X, com_radius, M, min_dist, d_min, k_1, k_2)
  G.optimize(local_max_iter=1000)
  X_star = G.X_star

  area_traj_fig, area_ax = plt.subplots()
  area_traj_fig.set_size_inches(7, 4)
  area_ax.plot(G.area_traj[1, :], G.area_traj[0, :]/M.area)
  area_ax.set_ylabel("Covered area [% of area of $\mathcal{F}$]")
  area_ax.set_yticklabels([f"{100*v:.1f}%" for v in area_ax.get_yticks()])
  area_ax.set_xlabel("Iteration")
  area_ax.xaxis.set_major_locator(MaxNLocator(integer=True))

  step_traj_fig, step_ax = plt.subplots()
  step_traj_fig.set_size_inches(7, 4)
  step_ax.set_ylabel("Step length")
  step_ax.set_xlabel("Iteration")
  step_ax.xaxis.set_major_locator(MaxNLocator(integer=True))

  for a in np.arange(X_star.shape[1]):
    step_ax.plot(np.arange(G.step_length_traj.shape[1]-1), G.step_length_traj[a, :-1])

  axs[1].set_title("Final configuration")
  plot_distr(X_star, M, com_radius, axs[1])

  print("---X_star---")
  print(X_star)

  config_fig.savefig(f"report/figs/{fignames}_distr.pdf", format="pdf", dpi=config_fig.dpi)
  area_traj_fig.savefig(f"report/figs/{fignames}_area_traj.pdf", format="pdf", dpi=area_traj_fig.dpi)
  step_traj_fig.savefig(f"report/figs/{fignames}_step_traj.pdf", format="pdf", dpi=step_traj_fig.dpi)

  if plot_steps:
    x_traj_fig, ex_traj_axs = plt.subplots(2, int((G.X_traj.shape[2]-1)/2))
    x_traj_fig.set_size_inches(12, 7)
    ex_traj_axs = ex_traj_axs.flatten()

    for i in np.arange(G.X_traj.shape[2]-1):
      X = G.X_traj[:, :, i]
      ex_traj_axs[i].set_title("Initial" if i == 0 else f"After iteration {i}")
      plot_distr(X, M, com_radius, ex_traj_axs[i])
    x_traj_fig.savefig(f"report/figs/{fignames}_x_traj.pdf", format="pdf", dpi=x_traj_fig.dpi)

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

  tinyworld = MissionSpace(np.array([
    [-0.75, -0.75],
    [0.75, -0.75],
    [0.75, 0.75],
    [-0.75, 0.75]
  ]))

  tinyworld2 = MissionSpace(np.array([
    [-0.75, -0.75],
    [0.75, -0.75],
    [0.75, 0.75],
    [-0.75, 0.75]
  ]), obstacles=[Obstacle.get_centered_box(0.75, 4)])

  bigworld = MissionSpace(np.array([
    [-5, -5],
    [5, -5],
    [5, 5],
    [-5, 5]
  ]))

  complexworld = MissionSpace(np.array([
    [-5, -5],
    [20, -5],
    [20, 10],
    [-5, 10],
  ]), obstacles=[
    Obstacle.get_vertical_wall(
      np.array([-3, -4]), 12
    ),
    Obstacle.get_hexagon(
      np.array([3, 5]), 3
    ),
    Obstacle.get_horizontal_wall(
      np.array([10, 3]), 4
    ) 
  ])

  complexworld2 = MissionSpace(np.array([
    [-5, -5],
    [5, -5],
    [5, 5],
    [-5, 5]
  ]), obstacles=[
    Obstacle.get_hexagon(
      np.array([3, 3]), 1
    ),
    Obstacle.get_hexagon(
      np.array([-4, 2]), 1
    ),
    Obstacle.get_hexagon(
      np.array([2, -2.5]), 2
    ),
    Obstacle.get_box(
      np.array([-2, -1]), 3
    ),
    Obstacle.get_vertical_wall(
      np.array([-4.5, -4]), 5
    )
  ])
  N_dots = 20
  com_radius = 3
  min_dist = 0.2
  d_min = 0.1

  X = np.array([
    [-5 + 0.1 + (-1)**(i)*0.05 for i in np.arange(N_dots)],  
    [-5 + (N_dots - i)*0.21 for i in np.arange(N_dots)]
  ])

  """
  Test for (k_1, k_2) = [(0, *), (1, 0.5), (1, 1), (1, 2), (2, 0.5), (2, 1), (2, 2)]
  """

  k_1 = 1
  k_2 = 1
  all_spawn(bigworld, X, com_radius, min_dist, d_min, k_1, k_2, fignames=f"bigworld_{N_dots}_agnt_k_1_{k_1}_k_2_{k_2}")


  