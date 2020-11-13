import numpy as np
from helpers import get_x_a, get_x_N_sub_a, get_visible_polygon, nCr, get_covered_polygon
from scipy.optimize import minimize
from itertools import combinations
from shapely.ops import unary_union

class DistrOpt():
  def __init__(self, X_N_init, com_radius, mission_space, min_dist, local_tol):
    self.X_N_init = X_N_init
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
    self.dynamic_constraints = lambda X, a: [
      {
        "type": "ineq",
        "fun": DistrOpt.no_same_pos_constraint,
        "args": (get_x_N_sub_a(X, a), self.min_dist)
      },
      {
        "type": "ineq",
        "fun": DistrOpt.min_two_neigh_constraint,
        "args": (get_x_N_sub_a(X, a), self.com_radius)
      }
    ]

  def optimize(self, local_max_iter = 500):
    visible_polys = np.array([
      get_visible_polygon(get_x_a(self.X_N_init, a), self.com_radius, self.mission_space)
    for a in np.arange(self.X_N_init.shape[1])])

    converged = np.zeros((self.X_N_init.shape[1]), dtype=bool)

    X = self.X_N_init

    print("OPTIMIZATION STARTED")

    self.area_traj = np.array([
      [get_covered_polygon(X, self.com_radius, self.mission_space).area],
      [0]
      ])
    iters = 0
    while not converged.all():
      iters += 1
      for a in np.arange(self.X_N_init.shape[1]):
        x_a_init = get_x_a(X, a).reshape(-1)
        ans = minimize(
          DistrOpt.objective,
          x_a_init,
          args = (X, a, visible_polys, self.com_radius, self.mission_space),
          constraints = self.static_constraints + self.dynamic_constraints(X, a),
          method="SLSQP",
          options={"maxiter": local_max_iter}
        )

        visible_polys[a] = get_visible_polygon(ans.x.reshape(2, 1), self.com_radius, self.mission_space)
        X = np.hstack((X[:, :a], ans.x.reshape(2, 1), X[:, a+1:]))
        converged[a] = np.linalg.norm(ans.x - x_a_init) <= self.local_tol
        print(f"Agent {a} step length: {np.linalg.norm(ans.x - x_a_init)}")
        if (self.mission_space.within_mission_space_bounds_constraint(get_x_a(X, a).reshape(2, 1)) < 0).any():
          print(f"Warning! Agent {a} breaking within mission space bounds constraint:")
          print(self.mission_space.within_mission_space_bounds_constraint(get_x_a(X, a).reshape(2, 1)))
        #if converged.all(): break
      self.area_traj = np.hstack((self.area_traj, np.array([
        [get_covered_polygon(X, self.com_radius, self.mission_space).area],
        [iters]
      ])))
    
    self.X_star = X
  
  @staticmethod
  def no_same_pos_constraint(s_i, s_bar_i, min_dist):
    s_i = s_i.reshape(2, 1)
    return (np.linalg.norm(s_i - s_bar_i, axis=0) - min_dist).reshape(-1)

  @staticmethod
  def min_two_neigh_constraint(s_i, s_bar_i, com_radius):
    s_i = s_i.reshape(2, 1)
    return len(np.where(np.linalg.norm(s_i - s_bar_i, axis=0) < 2*com_radius)[0]) - 2

  @staticmethod
  def objective(s_i, S, i, visible_polys, com_radius, mission_space):
    s_i = s_i.reshape(2, 1)
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
    return -unary_union(list(filter(lambda intersection: intersection != None, intersections))).area# + np.sum(np.exp(1-2.7*np.linalg.norm(s_i - S[:, b_i], axis=0)))