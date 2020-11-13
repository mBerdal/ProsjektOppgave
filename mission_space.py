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

  def plot(self, axis):
    x, y = self.exterior.xy
    axis.plot(x, y, color="gray")
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
        self.A_obstcls = obs_hull.equations[:, :2]
        self.b_obstcls = obs_hull.equations[:, 2:]
      else:
        self.A_obstcls = np.vstack((self.A_obstcls, obs_hull.equations[:, :2]))
        self.b_obstcls = np.vstack((self.b_obstcls, obs_hull.equations[:, 2:]))