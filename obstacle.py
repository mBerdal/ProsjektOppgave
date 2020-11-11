import numpy as np

class Obstacle():
  @staticmethod
  def get_centered_box(box_bounds, rel_size):
    return np.array([
      [-box_bounds/rel_size, -box_bounds/rel_size],
      [box_bounds/rel_size, -box_bounds/rel_size],
      [box_bounds/rel_size, box_bounds/rel_size],
      [-box_bounds/rel_size, box_bounds/rel_size]
    ])

  @staticmethod
  def get_four_symetrical_boxes(box_bounds, rel_size):
    return [
    np.array([
      [-box_bounds/rel_size, -box_bounds/rel_size],
      [box_bounds/rel_size, -box_bounds/rel_size],
      [box_bounds/rel_size, box_bounds/rel_size],
      [-box_bounds/rel_size, box_bounds/rel_size],
    ])-1,
    np.array([
      [-box_bounds/rel_size, -box_bounds/rel_size],
      [box_bounds/rel_size, -box_bounds/rel_size],
      [box_bounds/rel_size, box_bounds/rel_size],
      [-box_bounds/rel_size, box_bounds/rel_size],
    ])+1,
    np.array([
      [-box_bounds/rel_size, -box_bounds/rel_size],
      [box_bounds/rel_size, -box_bounds/rel_size],
      [box_bounds/rel_size, box_bounds/rel_size],
      [-box_bounds/rel_size, box_bounds/rel_size],
    ])+np.hstack((np.ones((4, 1)), -np.ones((4, 1)))),
    np.array([
      [-box_bounds/rel_size, -box_bounds/rel_size],
      [box_bounds/rel_size, -box_bounds/rel_size],
      [box_bounds/rel_size, box_bounds/rel_size],
      [-box_bounds/rel_size, box_bounds/rel_size],
    ])+np.hstack((-np.ones((4, 1)), np.ones((4, 1)))),
  ]

  @staticmethod
  def get_bottom_left_wall(box_bounds):
    return np.array([
      [-box_bounds + 1, -box_bounds+0.01],
      [-box_bounds + 1.1, -box_bounds+0.01],
      [-box_bounds + 1.1, 0],
      [-box_bounds + 1, 0]
    ])