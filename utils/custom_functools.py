# -*- coding: utf-8 -*-
"""
Module contents:

    #arguments
    #norecurse

@author daniel
"""

import traceback
import inspect

def arguments(depth=0):
    """Returns tuple containing dictionary of calling function's
       named arguments and a list of calling function's unnamed
       positional arguments.
       
       From:
       http://kbyanc.blogspot.co.uk/2007/07/python-aggregating-function-arguments.html
    """
    from inspect import getargvalues, stack
    posname, kwname, args = getargvalues(stack()[1+depth][0])[-3:]
    posargs = args.pop(posname, [])
    args.update(args.pop(kwname, []))
    return args, posargs
    
    


def norecurse(f):
    """ 
    silently return immediately to prevent impending recursion. This is rarely 
    a useful thing to do, but occasionally it might be.
    
    adapted taken from http://stackoverflow.com/a/7900380/2399799 """
    def func(*args, **kwargs):
        if len([l[2] for l in traceback.extract_stack() if l[2] == f.func_name]) > 0:
            return None
        else:
            return f(*args, **kwargs)
    return func
    



def append_docstring(*others):
    """
    Appends one or more other objects' docstrings to the decorated function.
    Title, location and args info is provided in addition to basic docstring.
    
    TODO: Compare the location of `other` to `foo` and only show the 
            neccessary suffix.  Also, try and get the args/kwargs and 
            type printed nicely like spyder does.
    """
    def decorator(foo):
        foo_doc = inspect.getdoc(foo)
        if foo_doc is None:
            foo_doc = "[no docstring]"
        
        for other in others:                        
            other_loc = inspect.getfile(other)
            other_doc = inspect.getdoc(other)
            if other_doc is None:
                other_doc = "[no docstring]"                   
            try:
                other_argspec = inspect.formatargspec(*inspect.getargspec(other))
                other_argspec = other_argspec[1:-1] # remove parens
                if other_argspec.startswith('self, '):
                    other_argspec = other_argspec[len('self, '):]
                other_argspec = "Arguments: ``" + other_argspec + "``\n\n" 
            except:
                other_argspec = ""
                
            foo_doc +=  "\n\n" + \
                        "Related: " + other.__name__ + "\n========\n\n"+ \
                        "Location: ``" + other_loc + "``\n\n" + other_argspec + \
                        other_doc
                    
        foo.__doc__ = foo_doc
        return foo
    return decorator
