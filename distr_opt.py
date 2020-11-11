import numpy as np
from helpers import get_s_i, get_s_bar_i, get_visible_polygon, nCr
from scipy.optimize import minimize
from itertools import combinations
from shapely.ops import unary_union

class DistrOpt():
  def __init__(self, S_init, com_radius, mission_space, min_dist, local_tol):
    self.S_init = S_init
    self.com_radius = com_radius
    self.mission_space = mission_space
    self.min_dist = min_dist
    self.local_tol = local_tol
    self.static_constraints = [
      {
        "type": "ineq",
        "fun": mission_space.within_mission_space_bounds_constraint,
      }
    ]

    if mission_space.num_obstacles > 0:
      self.static_constraints.append(
        {
          "type": "ineq",
          "fun": mission_space.outside_obstacles_constraint
        }
      )


  def optimize(self, local_max_iter = 500):
    w = self.S_init.shape[1]

    visible_polys = np.array([
      get_visible_polygon(get_s_i(self.S_init, i), self.com_radius, self.mission_space)
    for i in np.arange(w)])

    converged = np.zeros((w), dtype=bool)

    S = self.S_init

    while not converged.all():
      for i in np.arange(w):
        s_i_init = get_s_i(S, i).reshape(-1)
        ans = minimize(
          DistrOpt.objective,
          s_i_init,
          args = (S, i, visible_polys, self.com_radius, self.mission_space),
          constraints = self.static_constraints + [
            {
              "type": "ineq",
              "fun": DistrOpt.no_same_pos_constraint,
              "args": (get_s_bar_i(S, i), self.min_dist)
            }
          ],
          method="SLSQP",
          options={"maxiter": local_max_iter}
        )

        visible_polys[i] = get_visible_polygon(ans.x.reshape(2, 1), self.com_radius, self.mission_space)
        S = np.hstack((S[:, :i], ans.x.reshape(2, 1), S[:, i+1:]))
        converged[i] = np.linalg.norm(ans.x - s_i_init) <= self.local_tol
        print(f"Agent {i} converged: {converged[i]}, {np.linalg.norm(ans.x - s_i_init)}")
        if converged.all(): break
    
    self.S_star = S
  
  @staticmethod
  def no_same_pos_constraint(s_i, s_bar_i, min_dist):
    s_i = s_i.reshape(-1, 2).T
    return np.linalg.norm(s_i - s_bar_i, axis=0) - min_dist

  @staticmethod
  def objective(s_i, S, i, visible_polys, com_radius, mission_space):
    s_i = s_i.reshape(-1, 2).T
    S = S.reshape(-1, 2).T
    b_i_incl_i = np.where(np.linalg.norm(S - s_i, axis=0) < 2*com_radius)[0]
    b_i = b_i_incl_i[b_i_incl_i != i]
    if len(b_i) < 2:
      return 0
    V_s_i = get_visible_polygon(s_i, com_radius, mission_space)
    V_b_i = visible_polys[b_i]
    combs = list(combinations(V_b_i, 2))
    intersections = [None]*nCr(len(V_b_i), 2)
    i = 0
    for V_1, V_2 in combs:
      if V_1.intersects(V_2):
        intersection = V_1.intersection(V_2).intersection(V_s_i)
        for V_3 in V_b_i:
          if V_3 != V_1 and V_3 != V_2:
            intersection = intersection.difference(V_3)
        intersections[i] = intersection
        i += 1
    return -unary_union(list(filter(lambda intersection: intersection != None, intersections))).area