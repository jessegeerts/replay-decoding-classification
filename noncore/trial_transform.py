# -*- coding: utf-8 -*-

import numpy as np

from contextlib import contextmanager

def _make_rot_mat(theta):
    """simply use -theta to invert."""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

class TrialTransform(object):
    """this is a mixin for Trial"""
    
    @contextmanager
    def transformed(self, x_cm=None, y_cm=None, rot_deg=None, update_wh=True):
        """ Uses a hack to temporarily override the xy, w and h property. It also
            temporarily transform pos_shape using .rotate and .shift_xy.
            class/property hack is from: http://stackoverflow.com/a/31591589/2399799
           
           Use as:
               
               with my_trial.transformed(x_cm=4, y_cm=1, rot_deg=30):
                   my_trial.plot_spa_ratemap(t, c)
                   # etc.

        Note xy will be clipped to remain non-negative. Also, if update_wh is
        False, xy will be clipped to old w, h. When update_wh is true it is
        updated to ensure new xy fits within it (but negaitve values are sitll
        clipped to zero).
        
        Note that it is safe to clear pos cache within the
        block because the transformed xy is not actually
        part of the cache - it will only be lost when
        the with-block exits.  However, if you try and
        reload pos in a new way (e.g. after changing
        pos processor) the new changes will not be observed
        within the with-block...you could add extra methods
        here to raise exceptions but it's not really worth it.
        
        rotation is perforemd about centre  (w/2, h/2), then translation
        is perforemd.
        """
        if all(a is None for a in [x_cm, y_cm, rot_deg]):
            # no transform actually required
            yield 
            return

        original_class = self.__class__
        xy_trans = self.xy.copy()
        w_trans = self.w
        h_trans = self.h

        if rot_deg is not None:
            xy_trans -= np.array([[w_trans/2.], [h_trans/2.]])
            xy_trans = np.dot(xy_trans.T, _make_rot_mat(rot_deg/180.*np.pi)).T
            xy_trans += np.array([[w_trans/2.], [h_trans/2.]])

            
        if x_cm is not None:
            xy_trans[0] += x_cm
            
        if y_cm is not None:
            xy_trans[1] += y_cm
                
        np.maximum(xy_trans, 0, out=xy_trans)
        if update_wh:
            w_trans = max(np.max(xy_trans[0]), w_trans)
            h_trans = max(np.max(xy_trans[1]), h_trans)
        else:
            np.minimum(xy_trans, [[w_trans-0.001], [h_trans-0.001]], out=xy_trans)
            
        xy_trans.setflags(write=False)
                
        class TemporarilyTransformedTrial(original_class):
            xy = xy_trans
            w = w_trans
            h = h_trans
        
        try:
            self.pos_shape.rotate(rot_deg or 0)
            self.pos_shape.shift_xy(x_cm or 0, y_cm or 0)
            self.__class__ = TemporarilyTransformedTrial
            yield
        finally:
            self.pos_shape.shift_xy(-x_cm if x_cm else 0, -y_cm if y_cm else 0)
            self.pos_shape.rotate(-rot_deg if rot_deg else 0)            
            self.__class__ = original_class
            