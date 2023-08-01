# -*- coding: utf-8 -*-
"""
This is one of the options provided in trial_basic_analysis@get_spa_ratemap.
We consider it non-core though because it probably doesn't get used that much.

spk_pos_idx = self.tet_times(t=t, c=c, as_type='p')            
xy = self.xy
w = self._w
h = self._h
pos_samp_rate = self.pos_samp_rate

"""

import numpy as np
from numpy import  nan, newaxis as nax
from numpy_groupies import aggregate_np as aggregate
import numba

def get_ratemap_adaptive(xy, spk_pos_idx, w, h, spacing_cm, pos_samp_rate,
                         r_max=15, alpha=2500, mask_func=None, nodwell_mode='ma'):
    """
    WARNING: this seems to work, and parts of it have been tested reasonably
    well, but it's complicated so there may still be problems.
    
    mask_func is a function of the form: foo(x, y, fudge_cm). If it returns a
    truthy value the x-y point will be skipped.  It should permit broadcasting.
    
    This implements the method from Skags et al. 1996.  For a grid of points,
    i.e. ratemap bins, conceptually, we keep increasing a radius, r, until the
    following inequality is satisfied:
        
                            alpha
        n_spikes >  --------------------
                    dwell_time**2 * r**2
                    
    Note that we express the equation in terms of time rather than pos samps, this
    means the alpha value from Skags is 2500.  Once r is found, for the given
    point, we define the rate as simple n_spikes/dwell_time.
    
    The approach we take to optimizing is to put pos data into "buckets", so 
    we can then find easily find the superset of all the points within a given 
    radius of some location, by collecting a square ring of these buckets.  The
    actual implementation of the buckets is rather fiddly, so it's encapsulated
    in a special class.
    
    We also use jit-compilation (with numba) in order to make hard-to-vectorize
    loops go like native c-code.
    """
    # Compute the number of spikes for each pos sample
    
    n_pos = xy.shape[1]
    pos_spk_count = aggregate(spk_pos_idx, 1, size=n_pos)
    alpha *= pos_samp_rate**2 # this allows us to calculate in pos_samp units
    
    bucket_size = spacing_cm # these two do not have to be equal
    
    # bin the data into both c-order and f-order, grids...
    B = BucketGrid(xy, extra=pos_spk_count,
                   bucket_size=bucket_size, 
                   shape=np.ceil([h/bucket_size, w/bucket_size])) 

    r_max_bg = int(np.ceil(r_max/B.bucket_size))
    
    x_vals, y_vals = B.bin_centres()
    r_vals = np.empty((len(x_vals), len(y_vals)), dtype=float)
    rate_vals = np.empty((len(x_vals), len(y_vals)), dtype=float)
    r_vals.fill(nan), rate_vals.fill(nan)
    
    distsqr_and_count_buf = np.empty(n_pos, dtype=[('distsqr', np.float32), 
                                                   ('count', np.uint16)])
    
    is_outside = mask_func(x_vals[:, nax], y_vals[nax, :], fudge_cm=spacing_cm*1.5)
    
    for ii_x, x in enumerate(x_vals): 
        for ii_y, y in enumerate(y_vals): #for each point in the grid...
            if is_outside[ii_x, ii_y]:
                continue
            centre_xy = np.array([x, y])
            spikes_tot = 0
            dwell_tot = 0
            buf_len = 0

            for r_buckets in range(r_max_bg+1): # for concentric squares rings of increasing radius...
                
                # collect all the distsqrs and counts for the ring
                buf_len = get_distsqr_and_count_for_ring(B, centre_xy, r_buckets,
                                                         distsqr_and_count_buf,
                                                         buf_len)
                if buf_len == 0:
                    continue
                # sort by distsqr amd then see if anywhere down the list satisfies the inequality
                np.take(distsqr_and_count_buf[:buf_len],
                        np.argsort(distsqr_and_count_buf[:buf_len]['distsqr']),
                        out=distsqr_and_count_buf[:buf_len])
                flag, v1, v2 = find_first_satisfying_inequality(
                                            distsqr_and_count_buf[:buf_len],
                                            ((r_buckets+0.5)*bucket_size)**2, 
                                            alpha=alpha, 
                                            spikes_tot=spikes_tot,
                                            dwell_tot=dwell_tot)
                
                if flag == 0: # sucess
                    rate_vals[ii_x, ii_y] = v1
                    r_vals[ii_x, ii_y] = v2
                    break
                else: # failure, cannot satisfy inequality within valid radius
                    # update dwell and spike totals for the valid radius
                    # and discard the actual data within that valid radius
                    dwell_tot += v1
                    spikes_tot = v2
                    distsqr_and_count_buf[:buf_len-v1] = distsqr_and_count_buf[v1:buf_len]
                    buf_len = buf_len-v1

            #end r_buckets-loop
    #end x and y loops

    rate_vals *= pos_samp_rate
    if nodwell_mode == 'ma':
        return (np.ma.array(rate_vals, mask=np.isnan(rate_vals)), 
                np.ma.array(r_vals, mask=np.isnan(r_vals)))
    elif np.isscalar(nodwell_mode):
        rate_vals[np.isnan(rate_vals)] = nodwell_mode
        r_vals[np.isnan(r_vals)] = nodwell_mode
        return rate_vals, r_vals
    else:
        raise Exception("unrecognised nodwell_mode.")
    


@numba.jit(nopython=True)
def find_first_satisfying_inequality(distsqr_and_count, valid_distsqr, alpha, 
                                 spikes_tot, dwell_tot):
    """
    distsqr_and_count is only the ocupied part of the buffer, and it's
    sorted by distsqr.
    
    returns either:
        0, spike_count/pos_smaples, ball_radius     or...
        -1, idx_of_first_invalid, spikes_tot_valid
    """
    len_data = len(distsqr_and_count)
    for ii in range(len_data):
        if distsqr_and_count[ii].distsqr > valid_distsqr:
            return -1, ii, spikes_tot # valid_distsqr isn't big enough
        spikes_tot += distsqr_and_count[ii].count
        dwell_tot += 1
        if ii+1 < len_data and distsqr_and_count[ii].distsqr == distsqr_and_count[ii+1].distsqr:
            continue #  we need to get to the end of the bunch of equal-distant points
        if distsqr_and_count[ii].distsqr == 0:
            continue # cant divide by zero            
        denom = float(distsqr_and_count[ii].distsqr*(dwell_tot**2))
        if spikes_tot > alpha/denom:
            return 0, spikes_tot/float(dwell_tot), np.sqrt(distsqr_and_count[ii].distsqr)
    return -1, ii+1, spikes_tot

@numba.jit(nopython=True)
def sqeuclidean_and_copy_extra(xy_and_extra, cxy, out, out_offset):
    out_offset = int(out_offset)
    for ii in range(len(xy_and_extra)):
        dx = (xy_and_extra[ii, 0] - cxy[0])
        dy = (xy_and_extra[ii, 1] - cxy[1])
        out[ii+out_offset].distsqr = dx*dx + dy*dy
        out[ii+out_offset].count = xy_and_extra[ii, 2]
        

def get_distsqr_and_count_for_ring(B, centre_xy, r, buffer_, buf_len):
    """
    centre_xy is the coordinate of the centre in x,y space
    r is the radius in bucket grid bins
    buffer_ is a 2d array in which to store the result, and buf_len
    gives the strating write index into the buffer, i.e. the buffer may
    already have some data in it, which we have to append to.
    
    returns the new buf_len
    """
    c0, c1 = B.xy_to_bins(centre_xy)
    if r == 0:
        data = B.slice_d0_d1(c0, c1)
        sqeuclidean_and_copy_extra(data, centre_xy, buffer_, buf_len)
        return buf_len+len(data)
    else:
        # left side
        data = B.slice_d0start_d0stop_d1(c0-r, c0+r+1, c1-r)
        sqeuclidean_and_copy_extra(data, centre_xy, buffer_, buf_len)
        buf_len += len(data)

        # right side        
        data = B.slice_d0start_d0stop_d1(c0-r, c0+r+1, c1+r)
        sqeuclidean_and_copy_extra(data, centre_xy, buffer_, buf_len)
        buf_len += len(data)

        # top, without corners        
        data = B.slice_d0_d1start_d1stop(c0-r, c1-r+1, c1+r)
        sqeuclidean_and_copy_extra(data, centre_xy, buffer_, buf_len)
        buf_len += len(data)

        # bottom, without corners        
        data = B.slice_d0_d1start_d1stop(c0+r, c1-r+1, c1+r)
        sqeuclidean_and_copy_extra(data, centre_xy, buffer_, buf_len)
        buf_len += len(data)
        return buf_len
        
        
class BucketGrid(object):
    """
    Stores an array of 3-tuples: (x, y, extra), i.e. it's a 2d array.
    However, the x,y values are interpreted as coordinates, and are binned
    into a 2d "grid", this means there is an additional two dimentions, to
    give 4 in total. The "outer" two dimensions are handled
    by this class rather than by numpy. In fact, the entries in the grid
    contain 2d arrays of varying length.
    
    You can slice several columns along a row or several rows along a column,
    but not a rectangle of columns and rows.  The result is always of 
    shape nx3...
    
    BucketGrid[12, 30] 
    BucketGrid[10:15, 30]
    BucketGrid[19, 21:25]
    BucketGrid[20:23, 20:30] # not allowed

    Out of range indexing is dealt with differently to numpy:
    BucketGrid[-2:5, 10] # equivalent to BucketGrid[0:5, 10]
    BucketGrid[7:big_number, 10] # equivalent to Bucket[7:end, 10]
    BucketGrid[-2, 3:5] # returns 0x3 empty array
    BucketGrid[big_number, 3:5] # returns 0x3 empty array
    
    Behind the scenes, two copies of the data are kept, one with the rows
    of the grid contiguous and one with the columns of the grid contiguous.
    
    The .assert_slice_ok() method will check that the two copies agree and
    slicing operates as expected.
    
    """
    
    def __init__(self, xy, extra, bucket_size, shape):

        shape = shape.astype(int)
        self.bucket_size = float(bucket_size)
        self.shape = shape
        
        # get 1d bin index for each pos samp, do it for f and c order
        bin_idx = self.xy_to_bins(xy)
        bin_idx_c = np.ravel_multi_index(bin_idx, shape[::-1], order='C')
        bin_idx_f = np.ravel_multi_index(bin_idx, shape[::-1], order='F')
            
        # sort data by bin idx in f order and c order
        data = np.concatenate((xy.T, extra[:, nax]), axis=1)    
        sorting_idx_c = np.argsort(bin_idx_c)
        sorting_idx_f = np.argsort(bin_idx_f)
        self.data_c = data[sorting_idx_c, :]
        self.data_f = data[sorting_idx_f, :]
    
        # work out the (end+1) index for each bin, do it for f and c order...
        size = shape[0] * shape[1]    
        counts = np.bincount(bin_idx_c, minlength=size).reshape(shape, order='C')
        self.stop_c = np.cumsum(counts.ravel(order='C')).reshape(shape, order='C')
        self.stop_f = np.cumsum(counts.ravel(order='F')).reshape(shape, order='F')

        # used by __getitem__
        self.null_item = np.empty((0,3), dtype=self.data_c.dtype)
        
    def xy_to_bins(self, xy):
       return np.rint(xy/self.bucket_size).astype(int)
        
    def bin_centres(self):
        """ returns bin_centres_ax0, bin_centres_ax1 """
        return (np.arange(self.shape[0])*self.bucket_size,
                np.arange(self.shape[1])*self.bucket_size)
        
    def slice_d0_d1start_d1stop(self, d0, d1_start, d1_stop):
        if d0 < 0 or d0 >= self.shape[0]:
            return self.null_item
        d1_start = max(d1_start, 0)
        d1_stop = min(d1_stop, self.shape[1])
        stop_idx = self.stop_c[d0, d1_stop-1]
        if d1_start > 0:
            start_idx = self.stop_c[d0, d1_start-1]
        else:
            if d0 > 0:
                start_idx = self.stop_c[d0-1, -1]
            else:
                start_idx = 0
        return self.data_c[start_idx:stop_idx, :]
    
    def slice_d0start_d0stop_d1(self, d0_start, d0_stop, d1):
        if d1 < 0 or d1 >= self.shape[1]:
            return self.null_item
        d0_start = max(d0_start, 0)
        d0_stop = min(d0_stop, self.shape[0])
        stop_idx = self.stop_f[d0_stop-1, d1]
        if d0_start > 0:
            start_idx = self.stop_f[d0_start-1, d1]
        else:
            if d1 > 0:
                start_idx = self.stop_f[-1, d1-1]
            else:
                start_idx = 0
        return self.data_f[start_idx:stop_idx, :]
            
    def slice_d0_d1(self, d0, d1):
        if d0 < 0 or d0 >= self.shape[0] or d1 < 0 or d1 >= self.shape[1]:
            return self.null_item
        stop_idx = self.stop_c[d0, d1]
        if d1 > 0:
            start_idx = self.stop_c[d0, d1-1]
        else:
            if d0 > 0:
                start_idx = self.stop_c[d0-1, -1]
            else:
                start_idx = 0
        return self.data_c[start_idx:stop_idx, :]
                    
                    
    def __getitem__(self, key):
        d0, d1 = key
        if isinstance(d0, slice):
            # slice over multiple rows down one column
            assert not isinstance(d1, slice)
            return self.slice_d0start_d0stop_d1(d0.start, d0.stop, d1)
        elif isinstance(d1, slice):
            # slice over multiple columns along one row
            assert not isinstance(d0, slice)
            return self.slice_d0_d1start_d1stop(d0, d1.start, d1.stop)
        else:
            # single element
            return self.slice_d0_d1(d0, d1)
                    
    def assert_slice_ok(self):
        # bin (31, 22)
        a = self[31:32, 22].copy()
        b = self[31, 22:23].copy()
        c = self[31, 22].copy()
        a.sort(axis=0)
        b.sort(axis=0)
        c.sort(axis=0)
        assert np.all(a==b) and np.all(a==c)
        print("bin (31,22) is of shape {}x3 and is identically sliced via "
              "columns, rows, or as a single element."
                .format(len(a)))
        
        d = np.concatenate((self[32, 22], c), axis=0)
        e = self[31:33, 22].copy()
        d.sort(axis=0)                
        e.sort(axis=0)
        assert np.all(d==e)
        print("bins (31,22) and (32,22) have a combined shape {}x3 and are "
              "accessible with slice syntax [31:33, 22]."
                .format(len(e)))  

        assert all(x.flags.c_contiguous for x in [a,b,c,e])        
        print("all 2d arrays are c contiguous")
        



if __name__ == '__main__':  
    """ A test of sorts. Or perhaps better described as a debug check.
    """
    from tint import get_sample_trial
    a = get_sample_trial()
    B = BucketGrid(a.xy, np.arange(a.n_pos), 2.5, [44, 44])
    B.assert_slice_ok()
    r = 2
    c0, c1 = 14, 10
    v = np.zeros(a.n_pos)
    v[B[c0, c1][:,2].astype(int)] = 6
    v[B[c0-r:c0+r+1, c1-r][:, 2].astype(int)] = 2
    v[B[c0-r:c0+r+1, c1+r][:, 2].astype(int)] = 3
    v[B[c0-r, c1-r+1:c1+r][:, 2].astype(int)] += 4
    v[B[c0+r, c1-r+1:c1+r][:, 2].astype(int)] += 5
    a.plot_pos_alt(colors=v)
