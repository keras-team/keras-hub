import os
import numpy as np
import keras
import tensorflow as tf


def Polygon(coords):
    """
    Calculate the area of a polygon using the Shoelace formula.
    """
    coords = keras.ops.convert_to_tensor(coords,dtype="float32")
    x = coords[:, 0]
    y = coords[:, 1]

    x_next = keras.ops.roll(x, shift=-1, axis=0)
    y_next = keras.ops.roll(y, shift=-1, axis=0)

    area = 0.5 * keras.ops.abs(keras.ops.sum(x * y_next - x_next * y))
    return area

def shrink_polygan(poly,factor):
  """
  Shrink polygan inwards by a scaling its coordinated towards centroid
  """
  poly = keras.ops.convert_to_tensor(poly, dtype="float32")
  centroid = keras.ops.mean(poly, axis=0)  # Compute centroid

  # Correct scaling towards centroid
  shrinked_poly = centroid + (poly - centroid) * factor
  return shrinked_poly

def binary_search_smallest_width(poly):
  """
  The function aims maximum amount by which polygan can be shrunk by
  taking polygan's smallest width
  """
  if len(poly) < 3:
        return 0
    
  low, high = 0, 1  # Scale factor (1 = original size, 0 = collapsed to centroid)
  
  while high - low > 0.01:  # Precision threshold
      mid = (high + low) / 2
      mid_poly = shrink_polygan(poly, mid)
      area = Polygon(mid_poly)
      
      if area > 0.1:
          low = mid
      else:
          high = mid

  height = (low + high) / 2
  height = (low + high) / 2
  return int(height) if height >= 0.1 else 0

def project_point_to_line(x,u,v,axis=0):
    """
    Projects a point x onto the line defined by points u and v
    """
    x= keras.ops.convert_to_tensor(x,dtype="float32")
    u= keras.ops.convert_to_tensor(u,dtype="float32")
    v= keras.ops.convert_to_tensor(v,dtype="float32")

    n = v - u
    n = n / (keras.ops.norm(n, axis=axis, keepdims=True) + np.finfo(np.float32).eps)
    p = u + n * keras.ops.sum((x - u) * n, axis=axis, keepdims=True)
    return p

def project_point_to_segment(x,u,v,axis=0):
    """
    Projects a point x onto the line segment defined by points u and v
    """
    p = project_point_to_line(x, u, v, axis=axis)
    outer = keras.ops.greater_equal(keras.ops.sum((u - p) * (v - p), axis=axis, keepdims=True), 0)
    near_u = keras.ops.less_equal(keras.ops.norm(u - p, axis=axis, keepdims=True),keras.ops.norm(v - p, axis=axis, keepdims=True))
    o = keras.ops.where(outer, keras.ops.where(near_u, u, v), p)
    return o

def get_line_height(poly):
    """
    Get the height of the line defined by the polygan
    """
    return binary_search_smallest_width(poly) 

def line_segment_intersection(x, y, polygon):
    """
    Ray-casting algorithm to determine if a point is inside a polygon.
    https://medium.com/@girishajmera/exploring-algorithms-to-determine-points-inside-or-outside-a-polygon-038952946f87
    """
    inside = False
    num_vertices = len(polygon)
    for i in range(num_vertices):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % num_vertices]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside
  
def fill_poly(vertices, image_shape):
    """Fills a polygon using ray casting."""
    height, width = image_shape
    x = keras.ops.arange(width)
    y = keras.ops.arange(height)
    xx, yy = keras.ops.meshgrid(x, y)
    xx = keras.ops.cast(xx, "float32")
    yy = keras.ops.cast(yy, "float32")

    result = keras.ops.zeros((height, width), dtype="float32")
    xx_flat = keras.ops.reshape(xx, (-1,))
    yy_flat = keras.ops.reshape(yy, (-1,))

    inside_mask = [line_segment_intersection(xx_flat[i], yy_flat[i], vertices) for i in range(xx_flat.shape[0])]
    inside_mask_tensor = keras.ops.reshape(keras.ops.convert_to_tensor(inside_mask, dtype="bool"), (height, width))
    result = keras.ops.where(inside_mask_tensor, keras.ops.ones_like(result), result)
    return result

def get_mask(w, h, polys, ignores):
    """
    Generates a binary mask where:
    - Ignored regions are set to 0
    - Text regions are set to 1
    """
    mask = keras.ops.ones((h, w), dtype="float32")  
    
    for poly, ignore in zip(polys, ignores):
        poly = np.array(poly, np.int32)

        if poly.shape[0] < 3:
            print("Skipping invalid polygon:", poly)
            continue

        fill_value = 0.0 if ignore else 1.0
        poly_mask = fill_poly(poly, (h, w))

        if ignore:
          mask = keras.ops.where(poly_mask == 1.0, keras.ops.zeros_like(mask), mask)
        else:
          mask = keras.ops.maximum(mask, poly_mask)
    return mask

def get_region_coordinate(w, h, polys, heights, shrink):
    """
    Extract coordinates of regions corresponding to text lines in image using keras.ops.
    """
    label_map = keras.ops.zeros((h, w), dtype="int32")

    for line_id, (poly, height) in enumerate(zip(polys, heights)):
        if height > 0:
            shrinked_poly = shrink_polygan(poly, 1 - height * shrink)
            mask = fill_poly(shrinked_poly, (h, w))
            label_map = keras.ops.where(mask > 0, (line_id + 1) * keras.ops.ones_like(label_map), label_map)

    indices = keras.ops.convert_to_tensor(keras.ops.where(label_map > 0))
    if keras.ops.shape(indices)[0] == 0:
        return [np.zeros((0, 2), 'int32')] 
    
    label_map_flat = keras.ops.reshape(label_map, (-1,))
    flattened_indices = indices[..., 0] * w + indices[..., 1]
    region_labels = keras.ops.take(label_map_flat, flattened_indices)
    unique_labels, _ = tf.unique(region_labels)
    unique_labels = keras.ops.convert_to_tensor(unique_labels)

    regions_coords = []

    for label in unique_labels:
        region_idx = keras.ops.where(label_map == label)
        region_idx = keras.ops.convert_to_tensor(region_idx)

        coords = keras.ops.stack([region_idx[..., 1], region_idx[..., 0]], axis=-1)
        regions_coords.append(coords)

    return regions_coords

def get_coords_poly_projection(coords,poly):
      """
      This projects set of points onto edges of a polygan and return closest projected points
      """
      start_points = keras.ops.array(poly, dtype="float32")
      end_points = keras.ops.concatenate([keras.ops.array(poly[1:], dtype="float32"), 
                                          keras.ops.array(poly[:1], dtype="float32")], axis=0)
      region_points = keras.ops.array(coords, dtype="float32")

      projected_points = project_point_to_segment(
          keras.ops.expand_dims(region_points, axis=1),
          keras.ops.expand_dims(start_points, axis=0),
          keras.ops.expand_dims(end_points, axis=0),
          axis=2,
      )
      
      projection_distances = keras.ops.norm(
          keras.ops.expand_dims(region_points, axis=1) - projected_points, axis=2
      )

      indices = keras.ops.expand_dims(keras.ops.argmin(projection_distances, axis=1), axis=-1)
      best_projected_points = keras.ops.take_along_axis(projected_points, indices[..., None], axis=1)[:, 0, :]

      return best_projected_points

def get_coords_poly_distance_keras(coords, poly):
      """
      This function calculates distance between set of points and polygan
      """
      projection = get_coords_poly_projection(coords, poly)
      return keras.ops.linalg.norm(projection - coords, axis=1)

def get_normalized_weight(heatmap, mask,background_weight=3.0):
      """
      This function calculates normalized weight of heatmap
      """
      pos = keras.ops.greater_equal(heatmap, 0.5)
      neg = keras.ops.ones_like(pos, dtype="float32") - keras.ops.cast(pos, dtype="float32")
      pos = keras.ops.logical_and(pos, mask)
      neg = keras.ops.logical_and(neg, mask)
      npos = keras.ops.sum(pos)
      nneg = keras.ops.sum(neg)
      smooth = (keras.ops.cast(npos, dtype="float32") + keras.ops.cast(nneg, dtype="float32") + 1) * 0.05
      wpos = (keras.ops.cast(nneg, dtype="float32") + smooth) / (keras.ops.cast(npos, dtype="float32") + smooth)
      weight = np.zeros_like(heatmap)
      weight[keras.ops.cast(neg, dtype="bool")] = background_weight
      weight[keras.ops.cast(pos, dtype="bool")] = wpos
      return weight
      
