import numpy as np
from shapely.geometry import LineString
from shapely.ops import polygonize, unary_union

class IntersectionHelper():
  
  @staticmethod
  def __get_perimeter_intersection_angles(X, a, r):
    S_a = X[:, a].reshape(2, 1)
    in_range_indices = np.where(np.linalg.norm(S_a - X, axis=0) < 2*r)[0]
    in_range_indices = np.delete(in_range_indices, np.where(in_range_indices == a)[0])
    S_bar_a_in_range = X[:, in_range_indices]
    v_ij = S_bar_a_in_range - S_a
    alphas = np.arctan2(v_ij[1, :], v_ij[0, :]) + np.array([1, -1]).reshape(2, 1)*np.arccos(np.linalg.norm(v_ij, axis=0)/(2*r))
    alphas[alphas < 0] += 2*np.pi
    return alphas, in_range_indices

  @staticmethod
  def __get_sorted_angs_and_parent_list(X, a, r):
    angs, parents = IntersectionHelper.__get_perimeter_intersection_angles(X, a, r)
    angs_n_parents = np.vstack((
      angs.flatten(),
      np.tile(parents, 2),
      np.hstack((-np.ones((parents.shape)), np.ones((parents.shape))))
    ))
    return angs_n_parents[:, angs_n_parents[0,:].argsort()]

  @staticmethod
  def get_covered_parts(X, a, r, inter_ang_parent_dict, cov_degree, focus):
    cov = np.count_nonzero(np.linalg.norm(X[:, a].reshape(2, 1) + np.array([[r], [0]]) - np.delete(X, a, 1), axis=0) <= r)
    polys = []
    intersect_ops = []
    for i in np.arange(inter_ang_parent_dict[a].shape[1]):
      cov += inter_ang_parent_dict[a][2, i]
      if cov == cov_degree:
        point = X[:, a] + r*np.array([np.cos(inter_ang_parent_dict[a][0, i]), np.sin(inter_ang_parent_dict[a][0, i])])
        poly, inter = IntersectionHelper.__trav_covered(
          X, a, r,
          inter_ang_parent_dict,
          "ccw", point, point, focus
        )
        if not poly is None:
          polys.append(poly)
          intersect_ops.append(frozenset(inter))
    return polys, intersect_ops

  @staticmethod
  def __trav_covered(X, a, r, inter_ang_parent_dict, dir, seed_point, start_point, focus, polys = np.empty((2, 1)), operation_list = [], first = True):
    if np.isclose(seed_point, start_point).all() and not first:
      return polys[:, 1:], operation_list
      
    dist_to_focus  = np.linalg.norm(X[:, focus] - seed_point)
    if not a == focus and dist_to_focus > r and not np.isclose(dist_to_focus, r):
      return None, None
    angs = inter_ang_parent_dict[a][0, :]
    curr_index = np.argmin(
      np.linalg.norm(
        (X[:, a] - seed_point).reshape(2, 1) + r*np.array([np.cos(angs), np.sin(angs)])\
        , axis=0
      )
    )
    start_angle = inter_ang_parent_dict[a][0, curr_index]
    next_index = (curr_index + 1 if curr_index < inter_ang_parent_dict[a].shape[1]-1 else 0) if dir == "ccw" else curr_index-1
    end_angle = inter_ang_parent_dict[a][0, next_index]
    
    ang_diff = np.arctan2(np.sin(end_angle-start_angle), np.cos(end_angle-start_angle))
    num_points = np.max([100, int(np.ceil(100*r*np.abs(ang_diff)))])
    angs = np.linspace(start_angle, start_angle + ang_diff, num=num_points)
    arc = X[:, a].reshape(2, 1) + r*np.array([np.cos(angs), np.sin(angs)]).reshape(2, -1)

    next_a = int(inter_ang_parent_dict[a][1, next_index])
    
    next_dist_to_seed = np.linalg.norm(X[:, next_a] - seed_point)
    operation_list = operation_list + [("i", a)] if (dir == "ccw" and (not a == focus)) else operation_list
    return IntersectionHelper.__trav_covered(
      X, next_a, r,
      inter_ang_parent_dict,
      "ccw" if next_dist_to_seed <= r or np.isclose(next_dist_to_seed, r) else "cw",
      arc[:, -1],
      start_point,
      focus,
      np.hstack((polys, arc)),
      operation_list,
      False
    )

  @staticmethod
  def get_simple_covered_parts_and_ops(X, focus, r, cov_deg):
    ret_ops, ret_polys = [], []
    done_ops = set()
    inter_ang_parent_dict = {
      i: IntersectionHelper.__get_sorted_angs_and_parent_list(X, i, r) for i in np.arange(X.shape[1])
    }
    for i in np.arange(X.shape[1]):
      polys, ops = IntersectionHelper.get_covered_parts(X, i, r, inter_ang_parent_dict, cov_deg, focus)
      for j in np.arange(len(ops)):
        if not ops[j] in done_ops:
          done_ops.add(ops[j])
          ret_ops.append(ops[j])
          ls = LineString(np.c_[polys[j][0, :], polys[j][1, :]])
          lr = LineString(ls.coords[:] + ls.coords[0:1])
          mls = unary_union(lr)
          polygons = list(polygonize(mls))
          ret_polys.append(max(polygons, key=lambda p: p.area))

    return ret_ops, ret_polys