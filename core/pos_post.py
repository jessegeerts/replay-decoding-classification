# -*- coding: utf-8 -*-
import numpy as np
from numpy import newaxis as _ax
from warnings import warn
from scipy.ndimage.filters import convolve1d as sp_convolve1d
import matplotlib.pyplot as plt
import inspect
import numba

import sys

sys.path.append('.')

from utils.utils_circle_fit import fit_circle
from utils.custom_casting import str2float

# from replay_analysis.utils.numpy_groupies import aggregate_np as aggregate


# make Python 3 compatible:
try:
    # Python 2
    xrange
except NameError:
    # Python 3, xrange is now named range
    xrange = range


def _make_rot_mat(theta):
    """simply use -theta to invert."""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


@numba.jit(nopython=True)
def pos_post_jump_loop(x_vals, y_vals, is_bad, sqr_thresh, jump_over):
    """
    x_vals, y_vals : uint16 arrays with length n_pos
    is_bad : bool array with length n_pos
    sqr_thresh : square of max distance
    jump_over : bool, false-initialized array with length n_pos
    """
    n_pos = len(x_vals)

    # find the first good point
    for ii in xrange(n_pos):
        if not is_bad[ii]:
            break
    old_x, old_y = x_vals[ii], y_vals[ii]
    dt = 0

    for ii in xrange(ii + 1, n_pos):
        dt += 1
        dx = (x_vals[ii] - old_x)
        dy = (y_vals[ii] - old_y)
        if is_bad[ii] or (dx * dx + dy * dy) > (sqr_thresh * dt * dt):
            jump_over[ii] = True
        else:
            dt = 0
            old_x = x_vals[ii]
            old_y = y_vals[ii]


class PosProcessor(object):
    def __init__(self,
                 window_mode=1,
                 ppm_override=None,
                 swap_filter=True,
                 speed_filter=True,
                 max_speed_cm_per_s=4.0,  # speed filter in m/s
                 fit_shape=None,
                 circ_edge_percent=2,  # for matching to environment shape
                 rect_edge_percent=0.5,  # for matching to environment shape
                 circ_ang_res=15,
                 circ_rad_res=1,
                 circ_dwell_thresh=0.25 / 1200,  # fraction of trial spent in circle segment to count as occupied
                 circ_force_radius=None,  # value in cm, or None to use fitted value
                 smoothing_ms=400,  # this gives a 400ms smoothing window for pos averaging
                 swap_thresh=5,
                 fitted_padding=1.02,  # this is the factor used when padding around the fitted shape
                 fixed_w=None, fixed_h=None  # if a shape is fitted it will be centered within a box this size
                 ):
        """
            You supply a set of optional paramters here and then you get out a
            function of the form::
            
                process(led_pos, led_pix, pos_samp_rate, pos_header, n_leds, 
                        return_extra=False)
    
            That function will take in raw pos data and go through a number of
            steps to produce xy, and dir data.
            
            There are quite a few options going in to this factory function,
            best to check in the code for details.
        
            If you want to add your own stuff, it's best to stick to the
            pardaigm already in place, i.e. make a small function and put it
            into to pipieline somehwere in the process function,
            ideally with flags controlling its execution.
            
            Note that calcualting speed is pretty easy once you have a "nice" 
            set of xy values, so we leave it up to a separate function in Trial
            class.
            
            TODO: there is likely a confusion between x/y and w/h in various places...if it
            matters to you you'll have to fix it.
            
            TODO: there is still a problem with dir: may need to flip/rotate
                 and use correction offset from set file.
            
            TODO: there are a number of points where we loop over n_leds, although this isnt 
            going to be a bottleneck, it makes the code a bit messier, and can probably be
            vectorised into a single statement.
            
        """

        # store all kwargs as private attributes
        self._kwarg_names = inspect.getargvalues(inspect.currentframe()).args[1:]
        locals_ = locals()
        for k in self._kwarg_names:
            setattr(self, "_" + k, locals_[k])

    def __str__(self):
        return "PosProcessor with parameters: \n\t" + '\n\t'.join(
            k + ": " + str(getattr(self, "_" + k)) for k in self._kwarg_names)

    def __repr__(self):
        return str(self)

    def __call__(self, led_pos, led_pix, pos_samp_rate, pos_header, n_leds,
                 return_extra=False):
        """
        led_pos and led_pix are masked arrays.
        This function aplies a series of transformations, error masks, and LED swaps
        and then interpolates across the missing values.
        
        During processing, we store all the data stuff in a dummy class,
        _ProcessorState, this makes it easy to pass it around between
        each of the subfunctions.  Note that we aren't storing the data on
        the PosProcessor instance, because it should be considered state-less,
        i.e. it's a bit like an HTTP server processing a request.
        """

        state = _ProcessorState()  # this is holds the state of the data being processed

        state.n_leds = n_leds
        state.pos_header = pos_header
        state.pos_samp_rate = pos_samp_rate
        state.ppm = str2float(pos_header['pixels_per_metre']) if self._ppm_override \
                                                                 is None else self._ppm_override
        state.n_pos = led_pos.shape[0]
        if led_pos.ndim == 2:
            led_pos.shape = state.n_pos, 1, 2

        # Permute the dimensions of the arrays to match the alogrithms used here
        state.led_pos = np.transpose(led_pos, axes=[2, 1, 0])
        state.led_pix = np.transpose(led_pix, axes=[1, 0])
        # From this point forward we have the following format:
        # led_pos:  [x_and_y, n_leds, n_pos]
        # led_pix: [n_leds, n_pos]

        if np.ma.getmask(state.led_pos) is not np.ma.nomask:
            self._match_mask_for_xy(state)

        # From this point on we should maintain matched masking for (x, y) pairs 
        # (independantly for each LED)

        if self._window_mode == 1:
            self._window_coordinates(state)

        if np.any(state.led_pix) and n_leds == 2:
            if self._swap_filter:
                self._find_shrunken_big_led(state)
                self._find_crossovers(state)
                swap_is_needed = (state.big_led_is_shrunken & state.is_led_crossover)
                state.led_pos[:, :, swap_is_needed] = state.led_pos[:, ::-1, swap_is_needed]
                state.led_pix[:, swap_is_needed] = state.led_pix[::-1, swap_is_needed]

        if self._speed_filter:
            self._led_speed_filter(state)

        self._cast_pos_to_float(state)  # need to do this before converting to cm, or we will encur significant rounding

        self._pix_to_cm(state)

        state.env_shape = None
        if self._fit_shape is None:
            pass
        elif self._fit_shape in ("circle", "circ"):
            self._find_circle(state)
            self._window_coordinates_circ(state)
        elif self._fit_shape in ("rect", "rectangle"):
            self._find_rect(state)
            self._window_coordinates_rect(state)
        else:
            warn("PosProcessor fit_shape={}, is not a recognised fitting option."
                 .format(self._fit_shape))

        self._window_fixed_wh(state)

        self._get_led_weights(state)  # we do this just before interpolating across the mask

        self._interp_across_mask(state)

        self._smooth(state)  # this has to be done after interpolating across masked parts of led_pos

        self._combine_leds(state)  # this gets xy and dir (if more than 1 led)

        if return_extra:
            return state
        else:
            return state.xy, state.dir, 0, state.w, state.h, state.env_shape

    def _match_mask_for_xy(self, state):
        """ For each pair of x, y values in the led_pos (and there may be two such pairs per pos sample)
            it checks if either x or y is masked, and if so it masks the pair entirely.
        """
        bad_xy = np.any(state.led_pos.mask, axis=0)
        for led in range(state.n_leds):
            state.led_pos[:, led, bad_xy[led]] = np.ma.masked

    def _pix_to_cm(self, state):
        fac = (100. / state.ppm)
        state.led_pos *= fac;
        state.w = int(state.w * fac);
        state.h = int(state.h * fac);

    def _cast_pos_to_float(self, state):
        state.led_pos = np.ma.array(state.led_pos, dtype=np.single)

    def _window_coordinates(self, state):
        """ Constrain the led_pos to lie within the box defined by:
                window_min_x, window_max_x, window_min_y, window_max_y
            Any (x, y) pairs outside the region are masked.
            
            Not sure if its possible for data to lie outside this box, but whatever.
        """

        min_x = str2float(state.pos_header['window_min_x'])
        min_y = str2float(state.pos_header['window_min_y'])
        state.w = w = str2float(state.pos_header['window_max_x']) - min_x
        state.h = h = str2float(state.pos_header['window_max_y']) - min_y

        # It seems to be the case that min values are already subtracted
        # led_pos[0, :, :] -= min_x
        # led_pos[1, :, :] -= min_y

        upper_bound = np.array([w, h])[:, _ax, _ax]
        bad = (state.led_pos < 0) | (state.led_pos > upper_bound)
        bad = np.any(bad, axis=0)
        for led in range(state.n_leds):
            state.led_pos[:, led, bad[led]] = np.ma.masked

    def _find_rect(self, state):
        # put a rough rectangle around the data
        q = (self._rect_edge_percent, 100 - self._rect_edge_percent)
        box = np.nanpercentile(state.led_pos[:, 0].filled(np.nan),
                               q, axis=1)
        state.env_shape = _ShapeRect(box[0, 0], box[0, 1], box[1, 0], box[1, 1])

    def _window_coordinates_rect(self, state):
        """
        Shift min values/clip max values so that centre of rect is at centre 
        of window, and window edge is 2% of width beyond the edge of the circle.
        Values outside the rect are masked.
        """

        # Mask pos for each LED pos if x or y is outside box
        for led in range(state.n_leds):
            pos_is_bad = (state.led_pos[0, led] < state.env_shape.x1) | \
                         (state.led_pos[0, led] > state.env_shape.x2) | \
                         (state.led_pos[1, led] < state.env_shape.y1) | \
                         (state.led_pos[1, led] > state.env_shape.y2)
            state.led_pos[:, led, pos_is_bad] = np.ma.masked

        # shift x and y to centre the box, and define a larger w and h, using fitted_padding
        box_w = state.env_shape.x2 - state.env_shape.x1
        box_h = state.env_shape.y2 - state.env_shape.y1
        dx = state.env_shape.x1 - (self._fitted_padding - 1.0) * box_w / 2.0
        dy = state.env_shape.y1 - (self._fitted_padding - 1.0) * box_h / 2.0
        state.led_pos[0] -= dx
        state.led_pos[1] -= dy
        state.env_shape.shift_xy(-dx, -dy)
        state.w = box_w * self._fitted_padding
        state.h = box_h * self._fitted_padding

    def _find_circle(self, state):
        """ Fit circle to pos data algorithm by DM. Feb 2014. """

        # take all the non-nan pos samps from led 1
        xy = state.led_pos.data[:, 0, ~state.led_pos.mask[0, 0, :]]

        # First step is to approximate the centre of the circle by putting a rough
        # rectangle around the data
        q = (self._circ_edge_percent, 100 - self._circ_edge_percent)
        rect = np.nanpercentile(xy, q, 1)  # only use first LED
        xy_c = np.mean(rect, axis=1)

        # Now that we have an apprixmate centre, we divide the environment up
        # into small arcs, centred on this point.
        # The width and length of the arcs is defined by circ_rad_res and 
        # circ_ang_res respectively.
        xy -= xy_c[:, _ax]
        r = np.hypot(xy[0], xy[1])
        th = np.arctan2(xy[1], xy[0])  # not sure if that's right
        r_idx = (r / self._circ_rad_res).astype(int)
        th_idx = ((th + np.pi) * 180 / np.pi / self._circ_ang_res).astype(int)
        n_th = int(360.0 / self._circ_ang_res)
        np.clip(th_idx, 0, n_th - 1, out=th_idx)

        # For each angle we find the furthest out bin with more than circ_dwell_thresh fraction of the dwell time
        total_time_in_seg = aggregate((r_idx, th_idx), 1, size=[max(r_idx) + 1, n_th])
        total_time_in_seg = total_time_in_seg[::-1, :]  # revrse by r for argmax search...
        seg_r = np.argmax(total_time_in_seg > self._circ_dwell_thresh * state.n_pos, axis=0)
        seg_r = (total_time_in_seg.shape[0] - seg_r) * self._circ_rad_res

        # We now a set of points that roughly form a circle. So it should be easy to fit an actual circle..
        seg_th = np.arange(total_time_in_seg.shape[1]) * \
                 (self._circ_ang_res * np.pi / 180) - np.pi
        x = seg_r * np.cos(seg_th) + xy_c[0]
        y = seg_r * np.sin(seg_th) + xy_c[1]
        xc, yc, r = fit_circle(x, y)

        if self._circ_force_radius is not None:
            r = self._circ_force_radius

        state.env_shape = _ShapeCirc(xc, yc, r, x, y)

    def _window_fixed_wh(self, state):
        """
        if fixed_w and fixed_h are not none, the existing data will be shifted to centre
        it in the box defined by fixed_w and fixed_h. Outside this range it will be masked.
        w and h will be set to fixed_w and fixed_h.
        
        You can actually specify neither, one, or both of fixed_w and fixed_h.
        """
        if self._fixed_w is None and self._fixed_h is None:
            return

        pos_is_bad = np.zeros((state.n_leds, state.n_pos), dtype=bool)
        dx = dy = 0

        if self._fixed_w is not None:
            dx = (state.w - self._fixed_w) / 2.
            state.led_pos[0] -= dx
            state.w = self._fixed_w
            for led in range(state.n_leds):
                pos_is_bad[led] |= (state.led_pos[0, led] < 0) | \
                                   (state.led_pos[0, led] >= self._fixed_w)

        if self._fixed_h is not None:
            dy = (state.h - self._fixed_h) / 2.
            state.led_pos[1] -= dy
            state.h = self._fixed_h
            for led in range(state.n_leds):
                pos_is_bad[led] |= (state.led_pos[1, led] < 0) | \
                                   (state.led_pos[1, led] >= self._fixed_h)

        for led in range(state.n_leds):
            state.led_pos[:, led, pos_is_bad[led]] = np.ma.masked

        state.env_shape.shift_xy(-dx, -dy)

    def _window_coordinates_circ(self, state):
        """
        Shift min values/clip max values so that centre of circle is at centre 
        of window, and window edge is 2% of radius beyond the edge of the circle.
        Values outside the circle are masked.
        """
        pos_r_sqrd = (state.led_pos[0] - state.env_shape.cx) ** 2 + \
                     (state.led_pos[1] - state.env_shape.cy) ** 2
        pos_is_bad = pos_r_sqrd > state.env_shape.r ** 2
        for led in range(state.n_leds):
            state.led_pos[:, led, pos_is_bad[led]] = np.ma.masked

        dx = state.env_shape.cx - self._fitted_padding * state.env_shape.r
        dy = state.env_shape.cy - self._fitted_padding * state.env_shape.r
        state.led_pos[0] -= dx
        state.led_pos[1] -= dy
        state.w = 2.0 * self._fitted_padding * state.env_shape.r
        state.h = 2.0 * self._fitted_padding * state.env_shape.r
        state.env_shape.shift_xy(-dx, -dy)

    def _find_shrunken_big_led(self, state):
        """ For each sample, it checks if size of big light is closer to that of 
            the small light (as Z score). Returns a logical array, length n_pos.
        """
        led_pix = state.led_pix
        mean_npix = state.led_pix.mean(axis=1)
        std_npix = state.led_pix.std(axis=1)
        z11 = (mean_npix[0] - led_pix[0]) / std_npix[0]
        z12 = (led_pix[0] - mean_npix[1]) / std_npix[1]
        state.big_led_is_shrunken = z11 > z12

    def _find_crossovers(self, state):
        """
        Find where:
        
        # Big LED is significantly closer to small LED's previous position than 
        big LED's previous position. ..or..
        # Small LED is significantly closer to big LED's previous position than 
        small LED's previous position.
        
        I thinks that's basically what it's doing, right?
        """

        # Calculate Euclidean distances from one or other LED at fist time point 
        # to one or other LED at second time point
        dist12 = np.hypot(state.led_pos[0, 0, 1:] - state.led_pos[0, 1, :-1],
                          state.led_pos[1, 0, 1:] - state.led_pos[1, 1, :-1])
        dist11 = np.hypot(np.diff(state.led_pos[0, 0, :]),
                          np.diff(state.led_pos[1, 0, :]))
        dist21 = np.hypot(state.led_pos[0, 1, 1:] - state.led_pos[0, 0, :-1],
                          state.led_pos[1, 1, 1:] - state.led_pos[1, 0, :-1])
        dist22 = np.hypot(np.diff(state.led_pos[0, 1, :]),
                          np.diff(state.led_pos[1, 1, :]))

        state.is_led_crossover = np.zeros(state.n_pos, dtype=bool)
        state.is_led_crossover[:-1] = (dist11 - dist12 > self._swap_thresh) | \
                                      (dist22 - dist21 > self._swap_thresh)

    def _led_speed_filter(self, state):
        '''filters for impossibly fast tracked points, separately for each LED'''
        max_ppm_per_sample = self._max_speed_cm_per_s * state.ppm / state.pos_samp_rate
        max_ppms_sqd = max_ppm_per_sample ** 2
        x_vals = np.empty(state.n_pos, dtype=np.int16)
        y_vals = np.empty(state.n_pos, dtype=np.int16)
        for led in range(state.n_leds):
            jmp = np.zeros(state.n_pos, dtype=bool)
            x_vals[:] = state.led_pos.data[0, led]  # numba really doesn't like masked array slices
            y_vals[:] = state.led_pos.data[1, led]
            is_bad = np.any(state.led_pos.mask[:, led], axis=0)
            pos_post_jump_loop(x_vals, y_vals, is_bad, max_ppms_sqd, jmp)
            state.led_pos[:, led, jmp] = np.ma.masked

    def _interp_across_mask(self, state):
        '''        
        does a basic linear interpolation over missing values in the led_pos masked 
        array and returns the unmasked result
        '''
        is_missing = state.led_pos[0].mask  # we should have x-y missing matching
        is_ok = ~is_missing

        for led in range(state.n_leds):
            ok_data = state.led_pos.data[:, led, is_ok[led]]
            ok_idx, = is_ok[led].nonzero()
            missing_idx, = is_missing[led].nonzero()

            # separtely for x and y, take the ok_idx and the ok_data and fill in the data for missing_idx
            # (note that unlike matlab np's interp automatically extrapolates at the edges, using repeated value)
            state.led_pos.data[0, led, missing_idx] = \
                np.interp(missing_idx, ok_idx, ok_data[0])
            state.led_pos.data[1, led, missing_idx] = \
                np.interp(missing_idx, ok_idx, ok_data[1])

        state.led_pos = state.led_pos.data  # unmask the array

    def _get_led_weights(self, state):
        """ counts the number of masked pos in each led and defines the ratio 
        across LEDs as the weighting."""
        weights = np.ma.count(state.led_pos[0], axis=-1)
        state.weights = weights.astype(float) / np.sum(weights)

    def _combine_leds(self, state):
        if state.n_leds == 1:
            state.xy = state.led_pos[:, 0]
            state.dir = 'not available'
        else:
            state.xy = xy = np.empty([2, state.n_pos])
            weights = state.weights[_ax, :]
            xy[0] = weights.dot(state.led_pos[0])
            xy[1] = weights.dot(state.led_pos[1])
            # TODO: use correction from angle specified in settings
            # or estimate here based on fast run sections.
            state.dir = np.arctan2(np.diff(state.led_pos[0], axis=0),
                                   np.diff(state.led_pos[1], axis=0)).squeeze()

    def _smooth(self, state):
        """
        Does a boxcar smoothing of ``led_pos`` using a width of ``smoothing_ms``-miliseconds.        
        
        ``led_pos`` must be a non-masked array.
        """
        wid = int(state.pos_samp_rate * self._smoothing_ms / 1000)

        kern = np.ones(wid) / wid
        state.led_pos = sp_convolve1d(state.led_pos, kern, mode='nearest',
                                      axis=-1)
        # the "nearest" means repeated values at the ends


class _ProcessorState:
    """ a dummy class for holding position data state during processing """
    pass


class _Shape(object):
    """ A small class for holding info on a shape and alowing you to shift it around easily.
        
        Constructor
        ----
        
        * ``shape_type`` custom string for identifying the shape type
        
        * ``x_dict``, ``y_dict`` are dictionaries with attribute names as
          keys and scalars or np arrays as values.  Calling ``shift_xy(dx, dy)``
          updates all the values in each dict.
            
        * ``len_dict`` is similar to ``x_dict`` and ``y_dict`` but it specifies values
          that scale with length rather than specify coordinates, e.g. widths and radii.

        * All other key/values will be added to the _Shape instance as attributes
        
        
        Methods
        ----
        
        * ``shift_xy(dx, dy)`` updates the x and y values.

        * ``make_mask(w, h, bin_size_cm)`` - produces a mask of Trues outside the shape
          and False inside the shape.  Values are converted to bins using floor(val/bin_size_cm).
          w and h specify the full width and height of the region, ie. they must be
          converted to bin units to get the size of the mask.
        
        Subclasses
        ----
        ``_ShapeCirc``, ``_ShapeRect``
        
        TODO: implement the scaling thing. May also want a plot function.
        
        Example::
            
            >>> myCirc = _Shape("simpleCirc", x_dict={'cx':8}, y_dict={'cy':3}, 
                                somethingElse="hello")
            >>> print myCirc.cx, myCirc.somethingElse
             8 hello
            >>> myCirc.shift_xy(3, 2)
            >>> print myCirc.cx
             11     
    """

    def __init__(self, shape_type, x_dict=None, y_dict=None, len_dict=None,
                 **kwargs):
        """
        Note we dont check for duplicating of keys across dicts, this will cause a bug.
        """
        self.shape_type = shape_type
        self._rotation_deg = 0
        self._y_attrs = self._x_attrs = self._len_attrs = ()
        if x_dict is not None:
            self._x_attrs = x_dict.keys()
            self.__dict__.update(x_dict)
        if y_dict is not None:
            self._y_attrs = y_dict.keys()
            self.__dict__.update(y_dict)
        if len_dict is not None:
            self._len_attrs = len_dict.keys()
            self.__dict__.update(len_dict)
        if kwargs is not None:
            self.__dict__.update(kwargs)

    def shift_xy(self, dx, dy):
        for attr in self._x_attrs:
            v = getattr(self, attr)
            v += dx
            setattr(self, attr, v)

        for attr in self._y_attrs:
            v = getattr(self, attr)
            v += dy
            setattr(self, attr, v)

    def rotate(self, dth_degrees, cx=None, cy=None):
        if cx is not None or cy is not None:
            raise NotImplementedError("can only rotate about shape centre: "
                                      " use cx=None, cy=None")
        self._rotation_deg += dth_degrees

    def make_mask(self, w, h, bin_size_cm, inner_boundary=None):
        """ Normally the mask is True outside the shape and False inside.
        However, when inner_boundary > 0, we create a label matrix not
        a mask.  The region outside the shape is labeled 0, and inside the
        shape is labeled 1 in the outer anulus of width inner_boundary cm
        and labeled 2 inside that anulus.
    
        TODO: check this and all subclass versions are correct.       """
        w, h, bin_size_cm = float(w), float(h), float(bin_size_cm)
        m = np.zeros([int(np.ceil(w / bin_size_cm)),
                      int(np.ceil(h / bin_size_cm))], dtype=bool)
        if inner_boundary:
            raise NotImplementedError()
        return m

    def dist_to_boundary(self, xy):
        """ For an array of 2xn, returns the distance to the boundary for
        each value.        """
        raise NotImplemented()

    def is_outside(self, x, y, fudge_cm=None):
        """For a point, (x, y) in cm, it returns True if the point is outside
        the shape, otherwise False.  If fudge_cm is a number, then return 
        True if (x, y) is almost within the shape, using fudge_cm as a (rough)
        guide for what "close means".
        """
        return np.zeros(np.broadcast(x, y).shape, dtype=bool)

    def plot(self, w, h, bin_size_cm, inner_boundary=None):
        m = self.make_mask(w, h, bin_size_cm, inner_boundary=inner_boundary)
        if inner_boundary is None:
            m = np.ma.array(m, mask=~m)
        else:
            m = np.ma.array(m, mask=m == 0)
        plt.imshow(m.T, alpha=0.5, extent=[0, w, 0, h],
                   interpolation='nearest')

    def clip_region(self, **kwargs):
        """MPL Patch or Path for use with Artist.set_clip_path."""
        return None


class _ShapeCirc(_Shape):
    """ a circle defined by its centre and radius. 
        px and py are currently ignored.
        Note rotations don't matter for the circle."""

    def __init__(self, cx, cy, r, px, py):
        _Shape.__init__(self, "circ", x_dict={'cx': cx, 'px': px},
                        y_dict={'cy': cy, 'py': py}, len_dict={'r': r})

    def make_mask(self, w, h, bin_size_cm, inner_boundary=None):
        w, h, bin_size_cm = float(w), float(h), float(bin_size_cm)
        x = np.floor(abs((np.arange(0, w, bin_size_cm) + 0.5 - self.cx) / \
                         bin_size_cm)) * bin_size_cm
        y = np.floor(abs((np.arange(0, h, bin_size_cm) + 0.5 - self.cy) / \
                         bin_size_cm)) * bin_size_cm

        r_now = self.r
        bin_r = (x ** 2)[:, _ax] + (y ** 2)[_ax, :]
        m = bin_r >= r_now ** 2
        if inner_boundary is not None:
            inner_boundary = np.asarray(inner_boundary)
            inner_boundary.shape = (inner_boundary.size,)
            if np.any(inner_boundary < 0):
                raise Exception("inner_boundary cannot be negative.")
            r_now = -inner_boundary + r_now
            m = (~m).astype(np.uint8)
            for r_thresh in r_now:
                m += bin_r < r_thresh ** 2
        return m

    def dist_to_boundary(self, xy):
        """For a circle, the distance to the boundary is radius-dist_to_centre.
        """
        return self.r - np.hypot(xy[0] - self.cx, xy[1] - self.cy)

    def is_outside(self, x, y, fudge_cm=None):
        if fudge_cm is None:
            return (x - self.cx) ** 2 + (y - self.cy) ** 2 > self.r ** 2
        else:
            return (x - self.cx) ** 2 + (y - self.cy) ** 2 > (self.r + fudge_cm) ** 2

    def plot(self, w=None, h=None, bin_size_cm=None, show_mask=False,
             inner_boundary=None):
        """
        plots a circle using the cx, cy and r attribues.
        If show_mask is True, you must give w, h and bin_size_cm and it will
        also plot a mask that is transparent inside the shape and partially opaque
        outside.
        """
        if show_mask:
            _Shape.plot(self, w, h, bin_size_cm, inner_boundary=inner_boundary)
        plt.axis('equal')
        plt.gca().add_patch(plt.Circle((self.cx, self.cy), radius=self.r,
                                       color=[0, 0, 0], linewidth=2, linestyle='solid',
                                       alpha=0.8, fill=False))
        if inner_boundary is not None:
            plt.gca().add_patch(plt.Circle((self.cx, self.cy),
                                           radius=self.r - abs(inner_boundary),
                                           color=[0, 0, 0], linewidth=2, linestyle='solid',
                                           alpha=0.8, fill=False))
        plt.gca().relim()

    def clip_region(self, fc='none', ec='none', lw=0):
        """MPL Patch or Path for use with Artist.set_clip_path."""
        return plt.Circle((self.cx, self.cy), radius=self.r, facecolor=fc,
                          edgecolor=ec, linewidth=lw)


class _ShapeRect(_Shape):
    """
    A rectangle defined by the x cordinates of its left and right sides, and
    the y coordinates of its bottom and top.
    """

    def __init__(self, x1, x2, y1, y2):
        _Shape.__init__(self, "rect", x_dict={'x1': x1, 'x2': x2},
                        y_dict={'y1': y1, 'y2': y2})

    def make_mask(self, w, h, bin_size_cm, inner_boundary=None):
        if self._rotation_deg != 0:
            raise NotImplementedError("rect does not support rotation")

        bin_size_cm = float(bin_size_cm)
        m = _Shape.make_mask(self, w, h, bin_size_cm)

        m[:int(np.floor(self.x1 / bin_size_cm)), :] = True
        m[int(np.ceil(self.x2 / bin_size_cm)):, :] = True
        m[:, :int(np.floor(self.y1 / bin_size_cm))] = True
        m[:, int(np.ceil(self.y2 / bin_size_cm)):] = True
        if inner_boundary is not None:
            if inner_boundary < 0:
                raise Exception("inner_boundary cannot be negative.")
            m = (~m).astype(np.uint8)

            m[int(np.ceil((self.x1 + inner_boundary) / bin_size_cm)): \
              int(np.ceil((self.x2 - inner_boundary) / bin_size_cm)), \
            int(np.ceil((self.y1 + inner_boundary) / bin_size_cm)): \
            int(np.ceil((self.y2 - inner_boundary) / bin_size_cm))] += 1
        return m

    def dist_to_boundary(self, xy):
        if self._rotation_deg != 0:
            raise NotImplementedError("rect does not support rotation")

        return np.minimum( \
            np.minimum(np.abs(xy[0] - self.x1), np.abs(xy[0] - self.x2)),
            np.minimum(np.abs(xy[1] - self.y1), np.abs(xy[1] - self.y2)))

    def is_outside(self, x, y, fudge_cm=None):
        if self._rotation_deg != 0:
            raise NotImplementedError("rect does not support rotation")
        if fudge_cm is None:
            return ~((self.x1 < x) & (x < self.x2)
                     & (self.y1 < y) & (y < self.y2))
        else:
            # if it's almost inside, then say it is inside
            return ~((self.x1 - fudge_cm < x) & (x < self.x2 + fudge_cm)
                     & (self.y1 - fudge_cm < y) & (y < self.y2 + fudge_cm))

    def plot(self, w=None, h=None, bin_size_cm=None, show_mask=False,
             inner_boundary=None):
        """
        plots a circle using the cx, cy and r attribues.
        If show_mask is True, you must give w, h and bin_size_cm and it will
        also plot a mask that is transparent inside the shape and partially opaque
        outside.
        """
        ax = plt.gca()
        ax.add_patch(self.clip_region(lw=2, ec='k'))

        if show_mask:
            _Shape.plot(self, w, h, bin_size_cm, inner_boundary=inner_boundary)
        else:
            plt.xlim(0, w)
            plt.ylim(0, h)
        plt.axis('equal')

        if inner_boundary:
            if self._rotation_deg != 0:
                raise NotImplementedError("rect does not support rotation for inner bounday")
            plt.add_patch(plt.Rectangle((self.x1 + inner_boundary,
                                         self.y1 + inner_boundary),
                                        width=self.x2 - self.x1 - 2 * inner_boundary,
                                        height=self.y2 - self.y1 - 2 * inner_boundary,
                                        color=[0, 0, 0], linewidth=2, linestyle='solid',
                                        alpha=0.8, fill=False))

        ax.relim()

    def clip_region(self, fc='none', ec='none', lw=0):
        """MPL Patch or Path for use with Artist.set_clip_path."""

        if self._rotation_deg == 0:
            return plt.Rectangle((self.x1, self.y1), width=self.x2 - self.x1,
                                 height=self.y2 - self.y1, facecolor=fc,
                                 edgecolor=ec, linewidth=lw)
        else:
            cxy = np.mean([[self.x1, self.x2],
                           [self.y1, self.y2]], axis=1, keepdims=True).T
            points = np.dot(np.array([[self.x1, self.y1],
                                      [self.x1, self.y2],
                                      [self.x2, self.y2],
                                      [self.x2, self.y1]]) - cxy,
                            _make_rot_mat(self._rotation_deg / 180. * np.pi)) + cxy
            return plt.Polygon(points, closed=True, facecolor=fc,
                               edgecolor=ec, linewidth=lw)
