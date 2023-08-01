# -*- coding: utf-8 -*-
"""

Module contents:

    # TransformedDict - doesn't do much by itself, but you can sublcass it.
    # dict_ - this is a useable subclass of TransformedDict

    # LazyDict and LazyDictProperty


@author: daniel

"""

import collections
import time
from functools import partial

class TransformedDict(collections.MutableMapping):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys and returns a default
       value for mising keys.
       
       Note that this implemnts a dict interface but it does not
       subclass dict, so you cannot use isinstance(...,dict)

       From here with a couple of minor additions:
       http://stackoverflow.com/a/3387975/2399799
       """
    default_value = None
    
    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        return self.store.get(self.__keytransform__(key),self.default_value)

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.store[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return key
    
    def __contains__(self,key):
        return self.__keytransform__(key) in self.store
        
    def _repr_html_(self):
        return "<BR>".join(\
            ["<span style='background-color:#ccc;'>" + str(k) + ":</span> " + (v._repr_html_() if hasattr(v,'_repr_html_') else repr(v) ) for k,v in self.store.iteritems()]
        ) 
        
    def iteritems(self,disp=False,skip=(),enum=False):
        for ii,(k,v) in enumerate(self.store.iteritems()):
            if k in skip:
                continue
            if disp:
                print("[%d of %d | %s] '%s':" % (ii+1,len(self.store),time.strftime("%H:%M:%S"),k))
            yield ((ii,) if enum else () ) + (k,v)
        
        
class dict_(TransformedDict):
    """
    A dict-like class which treats all keys as lower case strings with a prefix.
    
    Example::    
    
    >>> D = dict_(g23='hello',g99='world',default='sorry',prefix='g')
    >>> D['g23']
    'hello'
    >>> D['23']
    'hello'
    >>> D[23]
    'hello'
    >>> D[81]
    'sorry'
    >>> 81 in D
    False
    >>> '23' in D and 23 in D and 'g23' in D
    True
    
        
    """    
    def __init__(self,prefix='',default=None,*args,**kwargs):
        self.prefix = prefix
        self.default_value = default
        TransformedDict.__init__(self,*args,**kwargs)
        
    def __keytransform__(self, key):
        key = str(key).lower()
        if key.startswith(self.prefix) is not True:
            key = self.prefix.lower() + key
        return key
    
    def _repr_html_(self):
        super_repr = super(dict_,self)._repr_html_()
        top_str = "<span style='text-decoration:underline;'>Dictionary with " + str(len(self))  + " items:</span><br>"
        if self.default_value is None:
            return top_str + super_repr
        else:
            return "<br>".join([\
                top_str + super_repr , \
                "<span style='background-color:#ccc;'>default:</span> " + str(self.default_value)  \
                ])
                
                
                

    
class LazyDictProperty(property):
    """
    Example useage
    --------------
    Suppose we have a class Man::
    
        from lazydict import LazyDictProperty
        
        class Man:
            @LazyDictProperty
            def knowledge(self,key):
                return "%s is in the eye of the beholder" % (key) 
            
        James = Man()
        print James.knowledge["beauty"]
        
    The idea is that in the ``knowledge`` method you will probably want to implement something 
    quite complicated. The benefit being that it only runs at the point you request a particular key.
    For convenience, you may want to store values in the ``self.knowledge._cache`` dict.
     
    Once you have created a LazyDictProperty, you have the opption of assigning an iterator. 
    To expand on the above example::
     
        class Man:
            @LazyDictProperty
            def knowledge(self,key):
                return "%s is in the eye of the beholder" % (key) 
            @knowledge.iter_function
            def knowledgeIter(self):
                return iter(["beauty","truth","reason"])
        
        James = Man()
        for idea in James.knowledge:
            print James.knowledge[idea]
 
     You can use round paren ``()`` to access the lazy dict instead of square ``[]``.  This 
     is another sense in which it is lazy.  In the round paren version you can pass a full
     set of args and kwargs. You can also store stuff in the ._cache dictionary of the LazyDict

     Implementation
     -------------
     When you wrap a method with the @LazyDictProperty you create a LazyDictProperty instance,
     which is a singleton for the class (i.e. in the example 'Man' has exactly one LazyDictProperty 
     instance). To enable the _cache (and make method binding a bit neater), each instance of
     the class (instances of 'Man' in the example) is given a _LazyDicts attribute which holds a
     dict of LazyDict instances built from each of the LazyDictProperties. Note that the _LazyDicts
     attribute and its keys/values do not exist until you access the relevant LazyDictProperty.
     
     TODO
     -------------------     
     It would be nice to do something more interesting with the cache, perhaps
     have a static method which can look up all instances of the class and clear cache.
     
     ...could get a bit complicated to decide what to clear...might need client code to
     specify a priority for each thing stored in the cache, where the priority will be applied
     globally.
    """
    def __init__(self,get_function):
        self._get_function = get_function
        def _default_iter_function(inst):
            raise TypeError("not iterable")
        self._iter_function = _default_iter_function
        
    def __get__(self,inst,type=None):
        if inst is None:
            return self# This is for autoreloading. TODO: not sure if this works absolutely perfectly...seems better now that we use property as base class not just object
            
        if not hasattr(inst,'_LazyDicts') or inst._LazyDicts is None:
            inst._LazyDicts = {}
        if self not in inst._LazyDicts:
            inst._LazyDicts[self] = LazyDict(partial(self._get_function,inst),partial(self._iter_function,inst))
        return inst._LazyDicts[self]
        
    def __set__(self,inst,val):
         raise AttributeError
        
    def iter_function(self,iter_function):
        self._iter_function = iter_function
        return iter_function
        
        
class LazyDict():
    """
    This is kind of similar to TransformedDict, but I wrote it several months earlier and for a different purpose.
    """
    def __init__(self,get_function,iter_function=None,canClearCache=False):
        self._get_function = get_function
        self._iter_function = iter_function
        self._cache = {} # this is provided for convenience, it doesn't do anything
        
    def __getitem__(self,key):
        return self._get_function(key)
        
    def __iter__(self):
        return self._iter_function()
        
    def __call__(self,*args,**kwargs):
        return self._get_function(*args,**kwargs)     
