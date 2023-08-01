# -*- coding: utf-8 -*-
import numpy as np


class TrialCaching(object):
    """ Mixin providing cache-management stuff """
    
    def __init__(self):
        self._cache_name_lists = {}
        
    def _cache_has(self, kind, names, key=None):
        """This is a statement of intent as well as a question.  
        Returns true if attribute ``name`` is avaialble and non-None.
        ``key` is optional, if provided it will trigger a check using ``name[key]``.
        Otherwise returns False.
        
        ``names`` can be a single str or tuple/list, in which case we loop over the 
        sequence reusing the single vales for kind (and key). We only return
        True if all names are available.
        
        IMPORTANT: if the attribute does not exist it will be created as
        ``None`` or ``{}`` if ``key`` is provided.  It will also be added to 
        the list of things which need to be cleared by _clear_cache for the
        given kind. You can use the standard 'pos', 'tet', 'eeg', 'set' or
        add your own kinds. When adding you own kinds you can provide a list
        of space-delimited dependencies in addition to the key itself, 
        e.g. kind="eeg theta" will be cleared when eeg or theta would be cleared."""

        names = (names,) if not isinstance(names, (list, tuple, set)) else names
            
        # update/create the list of names for the given kind
        if kind in self._cache_name_lists:
            self._cache_name_lists[kind].update(set(names))        
        else:
            self._cache_name_lists[kind] = set(names)
        
        # initialise any of the attributes if they are missing
        for name_ii in names:                 
            if not hasattr(self, name_ii):
                setattr(self, name_ii, None if key is None else {})
        
        # return False if any of the attributes are not already computed
        for name_ii in names:                 
            attr = getattr(self, name_ii)
            if attr is None or (key is not None and key not in attr):
                return False
        return True
        
    def _clear_cache(self, keep=(), drop=()):
        """specify either ``keep`` or ``drop``, or neither.
        They are tuples of strings corresponding to the keys of _cache_name_lists.
        See also _cache_has - useful for mixin classes."""
        if len(keep) and len(drop):
            raise ValueError("only keep or drop can be used, not both.")            
        if any(" " in k for k in keep + drop):
            raise NotImplementedError()
        
        if len(keep):
            kinds = [k for k in self._cache_name_lists.keys() if 
                                    all(kk in keep for kk in k.split(" "))]
        elif len(drop):
            kinds = [k for k in self._cache_name_lists.keys() if
                                    any(kk in drop for kk in k.split(" "))]
        else:
            kinds = self._cache_name_lists.keys()
            
        for k in kinds:
            for name in self._cache_name_lists[k]:
                if hasattr(self, name):
                    delattr(self, name) # this is safest and easiest way to do it
                    
    def _cache_verify_write_locks(self):
        """This is a testing utility which goes through everything in the
        _cache_name_list and checks to see if the write-flag is set to False.
        This is something specific to numpy arrays.  See caching notes in main
        readme.
        """
        print("Checking ndarrays for flags.writeable. will show 'WRITEABLE!!!' when True...\n")
        for name_list in self._cache_name_lists.values():
            for name in name_list:
                if not hasattr(self, name):
                    print(name, "(not available)")
                    continue
                
                attr = getattr(self, name)
                if isinstance(attr, np.ndarray):
                    if attr.flags.writeable:
                        print("###", name, "WRITEABLE!!!")
                    else:
                        print(name, "(locked ok)")
                elif hasattr(attr, 'items'): #strictly speaking this test isnt strong enough
                    ignored_count = 0
                    for k, v in attr.items():
                        if isinstance(v, np.ndarray):
                            if v.flags.writeable:
                                print("###", name, ":", k, "WRITEABLE!!!")
                            else:
                                print(name, ":", k, "(locked ok)")
                        else:    
                            ignored_count += 1
                    if ignored_count:
                        print(name, "(ignored", ignored_count, "of", len(attr), "items)")
                elif attr is None:
                    print(name, "(is None)")
                else:
                    print(name, "(not ndarray or dict-like)")
