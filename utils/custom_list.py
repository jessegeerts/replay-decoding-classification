# -*- coding: utf-8 -*-
"""
Module contents:

    # index_view
    # masked_list

@author: daniel
"""

import numpy as np
import logging

class index_view(object):
    """
    The class alows you to wrap an object, which contains one or more
    lists, and when you read or assign one of the lists you only read
    or assign to a single, pre-specified index.   For example::
    
    
        class Dummy():
            pass
        a_full = Dummy()
        a_full.thing = np.arange(5)*10
        a = index_view(a_full,3)
        print a.thing  # prints 30
        a.thing = -1
        print a.thing # prints -1
        print a_full.thing # prits [0,10,20,-1,40]
        
        
    The first time `thing` is requested, index_view passes it to a
    `new_attr_requested` handler.  The handler can do whatever it likes,
    such as create a new list etc. It then either returns an Exception,
    or a truthy value if the attribute is to be considered a list, or a falsy
    value if the attribute is to be returned as is, rather than indexed into.
    By default the handler returns `True`, indicating the attribute should be
    treated as a list.
    
    The `new_attr_requested` handler is set as a kwarg in the constructor
    
    The target object and index can be set in the constructor or directly 
    as `.__target` and `.__index`.    The direct method is useful if you are 
    iterating over a bunch of stuff and only want `new_attr_requested` to be 
    called once throughout the entire  interation process.
    
    Note that although we use the word `index`, the value can be any kind of key
    recognised by the sequences in question.
    
    Hopefully this now works with IPython's autoreload.  It seemed to be adding
    a prefic to attributes in a slightly conusing way.
    
    # TODO: pipe through additional magic methods as needed. 
            See http://stackoverflow.com/a/12311425/2399799     
    """    
    def __init__(self, target=None, index=None, new_attr_requested=lambda x: True):
        object.__setattr__(self, '__attr_is_indexable', {})
        object.__setattr__(self, '__new_attr_requested', new_attr_requested)
        object.__setattr__(self, '__index', index)
        object.__setattr__(self, '__target', target)
        
    def __getattr__(self, a):      
         # Hack to keep IPython autoreload happy...
        if a.endswith('__index'):
            return object.__getattribute__(self,'__index')
        elif a.endswith('__target'):
            return object.__getattribute__(self,'__target')
        
        attr_is_indexable = object.__getattribute__(self,'__attr_is_indexable')
        if a not in attr_is_indexable:
            attr_is_indexable[a] = object.__getattribute__(self,'__new_attr_requested')(a)

        t = getattr(self.__target, a)
        return t[self.__index] if attr_is_indexable[a] else t
        
    def __setattr__(self, a, v):
        # Hack to keep IPython autoreload happy...
        if a.endswith('__index'):
            return object.__setattr__(self, '__index', v)
        elif a.endswith('__target'):
            return object.__setattr__(self, '__target', v)

        attr_is_indexable = object.__getattribute__(self,'__attr_is_indexable')
        if a not in attr_is_indexable:
            attr_is_indexable[a] = object.__getattribute__(self,'__new_attr_requested')(a)

        if attr_is_indexable[a]:
            t = getattr(self.__target, a)
            t[self.__index] = v
        else:
            setattr(self.__target,a,v)
    def __repr__(self):
        return "index_view [index=" + str(self.__index) + "] of " + repr(self.__target)    
        
    """ pipe through magic methods to target """
    def __iter__(self):
        return iter(self.__target)
       
        
class masked_list():
    """
    This class presents a masked view of a full list.  If you ask for the
    kth element you will get the kth unmasked element.
    
    The mask must be a 1-length list containing either None or a numpy bool array, with Trues for the wanted elements.
    the purpose of boxing it in a 1-length list is so that you can switch the
    numpy mask array at a later date and have the new mask used..you can also update
    values in array without switching it compeltely.
    
    negative indexing and slicing should work fine.
    
    TODO: it would be more sensible to replace the 1-length iterable requirement
    with a proper Box class.
    """
    _warned_mask_objs = [] #used for reducing number of warnings..it't not bulett proof, but probably does the job...could be a list of weak refs or somehting but that's getting more complciated than we need for now
    
    def __init__(self,arr,dtype=np.dtype(object),mask=None):
        self.full_vals = np.array(arr,dtype=dtype)
        self._mask_boxed = mask
        
    @property
    def _mask(self):
        if len(self._mask_boxed) > 1:
            raise Exception("mask must be boxed in 1-length iterable")

        mask = self._mask_boxed[0]
        if mask is not None and (not isinstance(mask,np.ndarray) or mask.dtype is bool):
            raise Exception("mask must be numpy bool array")

        if mask is not None and len(mask) != len(self.full_vals):
            if any((v is mask for v in masked_list._warned_mask_objs)):
                logging.warn("mask is %d elements long but there are %d elements in full list." % (len(mask),len(self.full_vals)))
                masked_list._warned_mask_objs.append(mask)
            mask = np.concatenate((mask,np.zeros(len(self.full_vals)-len(mask),dtype=bool))) # append some Falses at the end

        return mask
        
    def __getitem__(self,k):
        if self._mask is None:
            return self.full_vals[k]
        else:
            return self.full_vals[self._mask][k]
    
    def __setitem__(self,k,v):
        """
        Note that this allows you do to::
        
            myList[3:6] = some_value
            
        but doing the following will produce different results to regualr numpy ::
        
            sliceOfMyList = myList[3:6]
            sliceOfMyList[:] = some_value
        
        In regular numpy you will have modified myList, but here you are only
        modifying a temporary copy of the slice of myList.
        Matching the behaviour of numpy is possible in theory I suppose, but more
        complicated.
        """
        if self._mask is None:
            self.full_vals[k] = v
        else:
            self.full_vals[self._mask.nonzero()[0][k]] = v
            
    def __len__(self):
        if self._mask is None:
            return len(self.full_vals)
        else:
            return sum(self._mask)
            
    def __iter__(self):
        """ Warning. Changing mask during iteration produces undefined results.
        Using this iterator should be more efficient that individaul key acces, because we
        only apply the mask once rather than once per access."""
        if self._mask is None:
            return iter(self.full_vals)
        else:
            return iter(self.full_vals[self._mask])
            
    def __repr__(self):
        return '!masked![' + ', '.join(map(str,iter(self))) + ']!masked: showing %d of %d elements!' % (len(self),len(self.full_vals))
            
    """
    using self[:] we get a temporary numpy array representing the masked list as it currently stands
    we can then use regular numnpy arithmetic comparison stuff...
    """
    def __eq__(self,other):
        return self[:] == other
    def __ne__(self,other):
        return self[:] != other
    def __lt__(self,other):
        return self[:] < other
    def __gt__(self,other):
        return self[:] > other
    def __le__(self,other):
        return self[:] <= other
    def __ge__(self,other):
        return self[:] >= other
            