import numpy as np
import json


def get_symmetry_transformations(model_info, max_sym_disc_step=None):
  """Returns a set of symmetry transformations for an object model.

  :param model_info: See files models_info.json provided with the datasets.
  :param max_sym_disc_step: The maximum fraction of the object diameter which
    the vertex that is the furthest from the axis of continuous rotational
    symmetry travels between consecutive discretized rotations.
  :return: The set of symmetry transformations.
  """
  # Discrete symmetries.
  
  trans_disc = [{'R': np.eye(3), 't': np.array([[0, 0, 0]]).T}]  # Identity.
  
  if 'symmetries_discrete' in model_info:
    for sym in model_info['symmetries_discrete']:
      sym_4x4 = np.reshape(sym, (4, 4))
      R = sym_4x4[:3, :3]
      t = sym_4x4[:3, 3].reshape((3, 1))
      trans_disc.append({'R': R, 't': t})

  # Combine the discrete and the discretized continuous symmetries.
  trans = []
  for tran_disc in trans_disc:
      trans.append(tran_disc)

  return trans





