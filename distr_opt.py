from matplotlib import pyplot as plt
import numpy as np
from numpy.lib.arraysetops import isin
from helpers import get_x_a, get_x_N_sub_a, get_visible_polygon, get_covered_polygon
from scipy.optimize import minimize
from shapely.ops import unary_union
from shapely.errors import TopologicalError
from shapely.geometry import MultiPolygon
from copy import deepcopy

class DistrOpt():
  def __init__(self, X_N_init, com_radius, mission_space, min_dist, d_min, k_1 = 1, k_2 = 1, local_tol = 10**(-2)):
    self.X_N_init = X_N_init
    self.com_radius = com_radius
    self.mission_space = mission_space
    self.min_dist = min_dist
    self.d_min = d_min
    self.k_1 = k_1
    self.k_2 = k_2
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
          "fun": mission_space.obs_test
        }
      )
    self.dynamic_constraints = lambda X_B_a: [
      {
        "type": "ineq",
        "fun": DistrOpt.no_same_pos_constraint,
        "args": (X_B_a, self.min_dist)
      },
      {
        "type": "ineq",
        "fun": DistrOpt.min_two_neigh_constraint,
        "args": (X_B_a, self.com_radius)
      },
    ]

  def __init_optimization(self):
    visible_polys = np.array([
      get_visible_polygon(get_x_a(self.X_N_init, a), self.com_radius, self.mission_space)
    for a in np.arange(self.X_N_init.shape[1])])
    
    X = self.X_N_init
    self.area_traj = np.array([
      [get_covered_polygon(X, self.com_radius, self.mission_space).area],
      [0]
    ])
    return X, visible_polys

  def __single_agent_optimization(self, a, X, visible_polys, local_max_iter):
    x_a_init = get_x_a(X, a)
    B_a_cup_a = np.where(np.linalg.norm(x_a_init - X, axis=0) <= 2*self.com_radius)[0]
    B_a = B_a_cup_a[B_a_cup_a != a]
    X_B_a = X[:, B_a]
    V_B_a = visible_polys[B_a]
    
    
    constraints = self.static_constraints + self.dynamic_constraints(X_B_a)
    if np.linalg.matrix_rank(X_B_a[:, 0].reshape(2, 1) - X_B_a[:, 1:]) < 2:
      constraints += [{
        "type": "ineq",
        "fun": DistrOpt.no_line_neigh,
        "args": (X_B_a, self.d_min)
      }]

    precomputed_intersections = DistrOpt.get_precomputed_intersections(V_B_a)

    x_a_init = deepcopy(x_a_init.reshape(-1))
    ans = minimize(
      lambda s, *args: DistrOpt.objective(s, *args),
      x_a_init,
      args = (X_B_a, precomputed_intersections, self.com_radius, self.k_1, self.k_2, self.mission_space),
      constraints = constraints,
      method="SLSQP",
      options={"maxiter": local_max_iter}
    )
    if ans.success:
      return ans.x, get_visible_polygon(ans.x.reshape(2, 1), self.com_radius, self.mission_space)
    print(f"Agent {a} rejected step due to {ans.message}")
    return x_a_init, visible_polys[a]


  def one_at_the_time_optimize(self, local_max_iter = 500):
    X, visible_polys = self.__init_optimization()
    x_a,_  = self.__single_agent_optimization(X.shape[1]-1, X, visible_polys, local_max_iter)
    X[0, X.shape[1]-1] = x_a[0]
    X[1, X.shape[1]-1] = x_a[1]
    return X

  def optimize(self, local_max_iter = 500):
    X, visible_polys = self.__init_optimization()
    self.X_traj = np.copy(X).reshape(*X.shape, 1)
    self.step_length_traj = np.zeros((X.shape[1], 2))
    converged = np.zeros((self.X_N_init.shape[1]), dtype=bool)

    print("OPTIMIZATION STARTED")
    print("covered area: ", get_covered_polygon(X, self.com_radius, self.mission_space).area)

    from helpers import plot_distr
    iters = 0
    while not converged.all() and iters < 100:
      iters += 1
      for a in np.arange(self.X_N_init.shape[1], dtype=int):
        x_a, V_a = self.__single_agent_optimization(a, X, visible_polys, local_max_iter)
        print(f"Agent {a} step length: {np.linalg.norm(X[:, a] - x_a)}")
        self.step_length_traj[a, iters] = np.linalg.norm(X[:, a] - x_a)
        converged[a] = np.linalg.norm(X[:, a] - x_a) <= self.local_tol
        visible_polys[a] = V_a
        X[0, a] = x_a[0]
        X[1, a] = x_a[1]

        if DistrOpt.min_two_neigh_constraint(x_a, get_x_N_sub_a(X, a), self.com_radius) < 0:
          print(f"Warning! Agent {a} has less than two neighbours")
      covered_area = get_covered_polygon(X, self.com_radius, self.mission_space).area
      print("covered area: ", covered_area, "iteration: ", iters)

      self.X_traj = np.append(self.X_traj, X.reshape(*X.shape, 1), axis=2)
      self.step_length_traj = np.hstack((self.step_length_traj, np.empty((X.shape[1], 1))))
      self.area_traj = np.hstack((self.area_traj, np.array([
        [covered_area],
        [iters]
      ])))
    
    self.X_star = X

  

  @staticmethod
  def no_same_pos_constraint(x_a, X_sub_a, min_dist):
    x_a = x_a.reshape(2, 1)
    return (np.linalg.norm(x_a - X_sub_a, axis=0) - min_dist).reshape(-1)

  @staticmethod
  def min_two_neigh_constraint(x_a, X_B_a, com_radius):
    x_a = x_a.reshape(2, 1)
    return X_B_a[:, np.linalg.norm(x_a - X_B_a, axis=0) <= 2*com_radius].shape[1] - 2

  @staticmethod
  def no_line_neigh(x_a, X_B_a, d_min):
    v = (x_a - X_B_a[:, 0]).reshape(2, 1)
    l = (X_B_a[:, 1] - X_B_a[:, 0]).reshape(2, 1)
    rejection = v - ((v.T@l)/(l.T@l))*l
    return np.linalg.norm(rejection)-d_min

  @staticmethod
  def get_precomputed_intersections(V_B_a):
    temp = []
    for i in np.arange(V_B_a.shape[0]):
      for j in np.arange(i+1, V_B_a.shape[0]):
        I = V_B_a[i].intersection(V_B_a[j])
        U = unary_union(np.delete(V_B_a, (i, j)))
        D = I.difference(U)
        if not D.is_empty:
          temp.append(D)
    return unary_union(temp)

  @staticmethod
  def get_area(x_a, precomputed_intersections, com_radius, mission_space):
    V_a = get_visible_polygon(x_a.reshape(2, 1), com_radius, mission_space)
    try:
      return (precomputed_intersections.intersection(V_a)).area
    except TopologicalError:
      A = 0
      for geom in precomputed_intersections.geoms:
        try:
          A += geom.intersection(V_a).area
        except TopologicalError as t:
          print("Intersects?: ", geom.intersects(V_a), type(geom))
          if geom.area >= 10**(-8):
            import matplotlib.pyplot as plt
            f, a = plt.subplots()
            a.plot(*geom.exterior.xy, "*")
            a.plot(*V_a.exterior.xy, "r--")
            print("Geom area: ", geom.area)
            f.show()
            plt.show()
            print("ALL WENT TO SHIT")
            exit(0)
      return A

  @staticmethod
  def close_dist_repell(x_a, X_B_a, k_2):
    return np.sum(np.exp(-k_2*np.linalg.norm(X_B_a - x_a.reshape(2, 1), axis=0)))

  @staticmethod
  def objective(x_a, X_B_a, precomputed_intersections, com_radius, k_1, k_2, mission_space):
    return -DistrOpt.get_area(x_a, precomputed_intersections, com_radius, mission_space) + k_1*DistrOpt.close_dist_repell(x_a, X_B_a, k_2)