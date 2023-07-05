import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import absl.logging
import datetime
import wandb

from tensorflow import keras
from tensorflow.keras import mixed_precision

tfd     = tfp.distributions
tfb     = tfp.bijectors

class SplineParams(tf.Module):

  def __init__(self, nbins=32, interval_width=2, range_min=-1,
               min_bin_width=1e-3, min_slope=1e-3):
    self._nbins = nbins
    self._interval_width = interval_width  # Sum of bin widths.
    self._range_min = range_min  # Position of first knot.
    self._min_bin_width = min_bin_width  # Bin width lower bound.
    self._min_slope = min_slope  # Lower bound for slopes at internal knots.
    self._built = False
    self._bin_widths = None
    self._bin_heights = None
    self._knot_slopes = None

  def __call__(self, x, nunits):
    if not self._built:
      def _bin_positions(x):
        out_shape = tf.concat((tf.shape(x)[:-1], (nunits, self._nbins)), 0)
        x = tf.reshape(x, out_shape)
        return tf.math.softmax(x, axis=-1) * (
              self._interval_width - self._nbins * self._min_bin_width
              ) + self._min_bin_width

      def _slopes(x):
        out_shape = tf.concat((
          tf.shape(x)[:-1], (nunits, self._nbins - 1)), 0)
        x = tf.reshape(x, out_shape)
        return tf.math.softplus(x) + self._min_slope

      self._bin_widths = tf.keras.layers.Dense(
        nunits * self._nbins, activation=_bin_positions, name='w')
      self._bin_heights = tf.keras.layers.Dense(
        nunits * self._nbins, activation=_bin_positions, name='h')
      self._knot_slopes = tf.keras.layers.Dense(
        nunits * (self._nbins - 1), activation=_slopes, name='s')
      self._built = True

    return tfb.RationalQuadraticSpline(
      bin_widths=self._bin_widths(x),
      bin_heights=self._bin_heights(x),
      knot_slopes=self._knot_slopes(x),
      range_min=self._range_min)

def spline_flow(nsplits = 1):
  splines = [SplineParams() for _ in range(nsplits)]
  stack = tfb.Identity()
  for i in range(nsplits):
    stack = tfb.RealNVP(5 * i, bijector_fn=splines[i])(stack)
  return stack