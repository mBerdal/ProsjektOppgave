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

  @staticmethod
  def get_vertical_wall(bottom_left_corner, length):
    return np.array([
      [bottom_left_corner[0], bottom_left_corner[1]],
      [bottom_left_corner[0] + 0.1, bottom_left_corner[1]],
      [bottom_left_corner[0] + 0.1, bottom_left_corner[1] + length],
      [bottom_left_corner[0], bottom_left_corner[1] + length]
    ])
  
  @staticmethod
  def get_horizontal_wall(bottom_left_corner, length):
    return np.array([
      [bottom_left_corner[0], bottom_left_corner[1]],
      [bottom_left_corner[0] + length, bottom_left_corner[1]],
      [bottom_left_corner[0] + length, bottom_left_corner[1] + 0.1],
      [bottom_left_corner[0], bottom_left_corner[1] + 0.1]
    ])
  
  @staticmethod
  def get_hexagon(center, side_length):
    angs = np.pi*((1/3)*np.arange(6) - 1/6)
    return (center.reshape(2, 1) + side_length*np.array([np.cos(angs), np.sin(angs)]).reshape(2, -1)).T