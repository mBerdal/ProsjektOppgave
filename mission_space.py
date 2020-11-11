from shapely.geometry import Polygon
from shapely.prepared import prep
from scipy.spatial import ConvexHull
import numpy as np

class MissionSpace(Polygon):
  def __init__(self, bounds, obstacles = []):
    super().__init__(bounds, obstacles)
    self.num_obstacles = len(obstacles)
    self.edge = bounds
    self.obstacles = obstacles
    self.prepped = prep(self)
    self.A_bounds, self.b_bounds, self.A_obstcls, self.b_obstcls = self.__get_ineq_constraint_matrices()

  def within_mission_space_bounds_constraint(self, s):
    h, = s.shape
    return -(np.kron(self.A_bounds, np.eye(int(h/2)))@s + np.tile(self.b_bounds.reshape(-1, ), int(h/2)))
  
  def outside_obstacles_constraint(self, s):
    h, = s.shape
    A = np.concatenate(self.A_obstcls, axis=0)
    b = np.concatenate(self.b_obstcls, axis=0)
    return np.max((np.kron(A, np.eye(int(h/2)))@s + np.tile(b.reshape(-1, ), int(h/2))))

  def plot(self, axis):
    x, y = self.exterior.xy
    axis.plot(x, y, color="gray")
    for interior in self.interiors:
      x, y = interior.xy
      axis.fill(x, y, fc="gray")

  def __get_ineq_constraint_matrices(self):
    F_hull = ConvexHull(self.edge)
    A_in_bounds = F_hull.equations[:, :2]
    b_in_bounds = F_hull.equations[:, 2:]
    A_in_obs_list = [None]*self.num_obstacles
    b_in_obs_list = [None]*self.num_obstacles
    for i in np.arange(self.num_obstacles):
      obs_hull = ConvexHull(self.obstacles[i])
      A_in_obs_list[i] = obs_hull.equations[:, :2]
      b_in_obs_list[i] = obs_hull.equations[:, 2:]
    return A_in_bounds, b_in_bounds, A_in_obs_list, b_in_obs_list