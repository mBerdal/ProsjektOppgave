from matplotlib import pyplot
from numpy.core.numerictypes import obj2sctype
from shapely.geometry import Polygon
from shapely.prepared import prep
from scipy.spatial import ConvexHull
import numpy as np

class MissionSpace(Polygon):
  def __init__(self, bounds, obstacles = []):
    super().__init__(bounds, obstacles)
    self.num_obstacles = len(obstacles)
    self.edges = bounds
    self.obstacles = obstacles
    self.__set_ineq_constraint_matrices()

  def within_mission_space_bounds_constraint(self, s):
    return -(self.A_bounds@(s.reshape(2, 1)) + self.b_bounds).reshape(-1)
  
  def outside_obstacles_constraint(self, s):
    return np.max(self.A_obstcls@(s.reshape(2, 1)) + self.b_obstcls)

  def obs_test(self, s):
    ans = np.zeros((len(self.A_obs_2, )))
    for i in np.arange(len(self.A_obs_2)):
      ans[i] = np.max(self.A_obs_2[i]@(s.reshape(2, 1)) + self.b_obs_2[i])
    return ans
  
  def plot(self, axis):
    axis.plot(*self.exterior.xy, color="gray")
    axis.fill(*self.exterior.xy, color="white")
    for interior in self.interiors:
      x, y = interior.xy
      axis.fill(x, y, fc="gray")

  def __set_ineq_constraint_matrices(self):
    F_hull = ConvexHull(self.edges)
    self.A_bounds = F_hull.equations[:, :2]
    self.b_bounds = F_hull.equations[:, 2:]
    for i in np.arange(self.num_obstacles):
      obs_hull = ConvexHull(self.obstacles[i])
      if i == 0:
        self.A_obs_2 = [obs_hull.equations[:, :2]]
        self.b_obs_2 = [obs_hull.equations[:, 2:]]
        self.A_obstcls = obs_hull.equations[:, :2]
        self.b_obstcls = obs_hull.equations[:, 2:]
      else:
        self.A_obstcls = np.vstack((self.A_obstcls, obs_hull.equations[:, :2]))
        self.b_obstcls = np.vstack((self.b_obstcls, obs_hull.equations[:, 2:]))
        self.A_obs_2.append(obs_hull.equations[:, :2])
        self.b_obs_2.append(obs_hull.equations[:, 2:])

if __name__ == "__main__":
  import mapbox_earcut as earcut
  import matplotlib.pyplot as plt
  m = np.array([
    [-1, -1],
    [1, -1.5],
    [1, 1],
    [-1, 1]
  ])
  O = [np.array([
    [-0.5, -0.5],
    [0.5, -0.5],
    [0.5, 0.5],
    [-0.5, 0.5]
  ])]
  M = MissionSpace(m,
  obstacles=O)
  print(m.shape)
  P = np.vstack((m, O[0]))
  res = earcut.triangulate_float32(P, [m.shape[0], m.shape[0] + O[0].shape[0]])
  print(res)
  for i in np.arange(0, len(res), 3):
    T = np.vstack((P[res[i:i+3]], P[res[i]]))
    plt.plot(*T.T, zorder=200, color="green")
    print(res[i:i+3])
  plt.show()