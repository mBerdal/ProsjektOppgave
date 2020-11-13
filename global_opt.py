import numpy as np
from helpers import get_x_a, get_x_N_sub_a, get_covered_polygon
from itertools import combinations
from scipy.optimize import minimize

class GlobalOpt():
  def __init__(self, S_init, com_radius, mission_space, min_dist):
    self.S_init = S_init
    self.com_radius = com_radius
    self.mission_space = mission_space
    self.min_dist = min_dist
    self.constraints = [
      {
        "type": "ineq",
        "fun": mission_space.within_mission_space_bounds_constraint
      },
      {
        "type": "ineq",
        "fun": GlobalOpt.no_same_pos_constraint,
        "args": (min_dist, )
      }
    ]
    if mission_space.num_obstacles > 0:
      self.constraints.append(
        {
          "type": "ineq",
          "fun": mission_space.outside_obstacles_constraint
        }
      )

  def optimize(self, max_iter = 500):
    _, w = self.S_init.shape
    ans = minimize(
      GlobalOpt.objective,
      self.S_init.T.reshape(2*w, ),
      args = (self.com_radius, self.mission_space),
      constraints=self.constraints,
      method="SLSQP",
      options={"maxiter": max_iter}
    )
    print(ans.message)
    self.S_star = ans.x.reshape(-1, 2).T

  @staticmethod
  def objective(s, com_radius, mission_space):
    s = s.reshape(-1, 2).T
    return -get_covered_polygon(s, com_radius, mission_space).area + GlobalOpt.close_dist_repell(s, 2)

  @staticmethod
  def no_same_pos_constraint(s, min_dist):
    s = s.reshape(-1, 2).T
    w = s.shape[1]
    return np.array([np.linalg.norm(s[:, comb[0]] - s[:, comb[1]]) - min_dist for comb in list(combinations(np.arange(w), 2))])

  @staticmethod
  def close_dist_repell(s, n):
    _, w = s.shape
    all_norms = np.array([np.linalg.norm(get_x_a(s, i) - get_x_N_sub_a(s, i), axis=0) for i in np.arange(w)])
    n_nearest_indices = np.argpartition(all_norms, n-1, axis=1)[:, :n]
    return np.sum(np.exp((1-2.7*np.sum([all_norms[i, n_nearest_indices[i]] for i in np.arange(w)], axis=1))))
