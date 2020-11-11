import numpy as np
from math import factorial
from itertools import combinations
from shapely.ops import unary_union, cascaded_union
from shapely.geometry import Point, MultiPoint, Polygon
from descartes.patch import PolygonPatch

def atan(y, x):
  return np.arctan2(y, x) + np.where(y < 0, 2*np.pi, 0)

def ssa(a1, a2):
  return np.min(np.abs([a1-a2, a1-a2+2*np.pi, a1-a2-2*np.pi]), axis=0)

def nCr(n, r):
  return int(factorial(n)/(factorial(r)*factorial(n-r)))

def get_s_i(s, i):
  return s.reshape(2, -1)[:, i].reshape(2, -1)

def get_s_bar_i(s, i):
  temp = s.reshape(2, -1)
  _, w = temp.shape
  return s[: ,np.r_[0:i, i+1:w]]

def get_edge_intersections(s, com_radius, obstacle, drop_vertices = []):
  check_vertices = np.array([i for i in np.arange(obstacle.shape[0]) if not i in drop_vertices])

  shifted_obstacle = obstacle[np.hstack(([-1], np.arange(0, obstacle.shape[0]-1))), :]
  v21 = (obstacle - shifted_obstacle)[check_vertices, :]
  v1s = (shifted_obstacle - np.tile(s, obstacle.shape[0]).reshape(obstacle.shape[0], 2))[check_vertices, :]
  dotpr211s = np.sum(v21*v1s, axis=1)
  dotpr2121 = np.sum(v21*v21, axis=1)
  dotpr1s1s = np.sum(v1s*v1s, axis=1)
  discriminant = np.power(2*dotpr211s, 2) - 4*dotpr2121*(dotpr1s1s - np.power(com_radius, 2))
  valid, = np.where(discriminant >= 0)

  d1 = -dotpr211s[valid]/dotpr2121[valid]
  d2 = (1/(2*dotpr2121[valid]))*np.sqrt(discriminant[valid])
  lambdas = (d1.reshape(-1, 1) + d2.reshape(-1, 1)*np.array([1, -1]))
  sols = np.where((lambdas > 0) & (lambdas < 1), lambdas, np.nan).reshape(-1)

  obs = np.repeat(obstacle[check_vertices, :][valid, :].T, 2, axis=1)
  shifted_obs = np.repeat(shifted_obstacle[check_vertices, :][valid, :].T, 2, axis=1)
  temp = shifted_obs + sols*(obs-shifted_obs)
  return temp[~np.isnan(temp)].reshape(2, -1)


def get_invisible_polygon(s, com_radius, obstacle):
  in_range_indices, = np.where(np.linalg.norm(obstacle.T - s.reshape(2, -1), axis=0) <= com_radius)
  numb_in_range_vertices = in_range_indices.shape[0]
  interest_pts = None
  if numb_in_range_vertices == 0:
    interest_pts = get_edge_intersections(s, com_radius, obstacle)
  elif numb_in_range_vertices < obstacle.shape[0]:
    interest_pts = np.hstack((
      obstacle.T[:, in_range_indices],
      get_edge_intersections(s, com_radius, obstacle, in_range_indices)
    ))
  else:
    interest_pts = obstacle.T

  if interest_pts.shape[1] == 0:
    return Point()

  vecs = interest_pts - s.reshape(2, 1)
  vec_angles = atan(vecs[1, :], vecs[0, :])
  combs = np.array(list(combinations(vec_angles, 2)))

  """
  angs: angles (wrpt. x-axis) of vectors from s to interest_pts
  which have the largest angle between them
  """

  angs = combs[np.argmax(ssa(combs[:, 0], combs[:, 1]))]

  min_ang = min(angs)
  max_ang = max(angs)
  mid_ang = (max_ang + min_ang)/2 if max_ang-min_ang < np.pi else max_ang + min_ang

  pts = np.tile(s, 4).reshape(4, 2) + com_radius*np.array([
    [np.cos(min_ang), np.sin(min_ang)],
    [np.cos(max_ang), np.sin(max_ang)],
    [np.cos(mid_ang) + np.cos(min_ang), np.sin(mid_ang) + np.sin(min_ang)],
    [np.cos(mid_ang) + np.cos(max_ang), np.sin(mid_ang) + np.sin(max_ang)]
  ])

  return MultiPoint(np.vstack((pts, obstacle))).convex_hull

def get_visible_polygon(s, com_radius, mission_space):
  invisible_multipoly = unary_union([
    get_invisible_polygon(s, com_radius, obs) for obs in mission_space.obstacles
  ])
  return Point(s).buffer(com_radius).intersection(mission_space).difference(invisible_multipoly)

def get_covered_polygon(s, com_radius, mission_space):
  _, w = s.shape
  polys = [
    get_visible_polygon(s[:, i], com_radius, mission_space) for i in np.arange(w, dtype=int)
  ]
  i = 0
  intersections = [None]*nCr(w, 3)
  for p1, p2, p3 in combinations(polys, 3):
    intersections[i] = p1.intersection(p2).intersection(p3)
    i += 1
  return cascaded_union(intersections)

def plot_cover_aux(cover_poly, ax, clr="blue", alpha=0.2):
  if isinstance(cover_poly, Polygon):
    ax.add_patch(PolygonPatch(cover_poly, color=clr, alpha=alpha))
  else:
    for geom in cover_poly.geoms:
      ax.add_patch(PolygonPatch(geom, color=clr, alpha=alpha))