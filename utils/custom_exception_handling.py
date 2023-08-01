# -*- coding: utf-8 -*-
"""
Module contents:

    # ExceptionAggregator, which makes use of _StateCM   
    # ignored
       

@author: daniel
"""

from __future__ import print_function

import sys
import warnings
from traceback import format_list, extract_tb, extract_stack
from itertools import groupby
from contextlib import contextmanager
import re 

try:
    from IPython.display import display
except Exception:
    display = print        

"""
try:
    raise NotImplementedError
    import IPython.core.ultratb as ipy_tb
    format_list2 = lambda x,m,ls: ''.join(ipy_tb.ListTB().structured_traceback(x,m,ls))\
                    .replace('<','&lt;').replace('>','&gt;').replace('\n','<BR>')
except Exception:
"""
format_list2 = lambda x,m,ls: '{ SPAN_C }' + \
                                (' '.join(format_list(ls))).replace('<','&lt;').replace('>','&gt;').replace('\n','<BR>').replace(' ','&nbsp;') +\
                                 '{/ SPAN }{ SPAN_B }' + x + ': {/ SPAN }' + str(m)

class ExceptionAggregator(object):
    """
    Usage::
    
    with ExceptionAggregator() as xh:
        for ii, s in enumerate(something):
            with xh.state("[" + str(ii) + "] " + s ):
                pass #do something here which might raise exceptions and/or warnings
        
    The inner with-block will catch exceptions and warnings silently, but when the outer with-
    block is exited, it will print a log of all the exceptions and warnings that have occured.
    (The outer with-block doesn't catch exceptions or warnigns, only the inner block does).
    The log will give the state string provided to ``xh.state``.    
    
    Note that by default warnings are usually only emited once per python
    instance, per message/type...you can change this in the warnings module if you want.    
    While that is probably good behaviour normally, you may want to record all
    occurrences of a warning here. (TODO: provide an easy way to change that here.)
    
    You don't have to use this ExceptionAggregator in the double or even single with-block mode 
    you can directly call ``xh.log(state_str)`` to record the most recent
    exception together with your custom state string.  And then use ``str(xh)``
    to get the log.  One benefit of using the inner with-block syntax is that
    you get warnings handled which you cant do with a simple try-catch block.
    
    """
    exception_symbol = '*'
    warning_symbol = '^'
    
    def __init__(self):
        self._stacks = [] # this is going to hold the stack stubs from try-catch blocks and warngins.showwarning
        self._ctx_stack = [] # this holds the stack at the point we enter the context, we use this for trimming the top off warning stacks
        
    def state(self,*args):
        """
        Create a context manager for the nested with-block.
        This nested c-m will call ``.log`` and may directly add to ``._stack``.
        """
        return _StateCM(self, ' '.join(str(a) for a in args))
        
    def log(self,state):
        sys.stdout.flush()
        print(self.exception_symbol,end='')
        sys.stdout.flush()
        self._stacks.append((state,sys.exc_value,sys.exc_type,tuple(extract_tb(sys.exc_info()[2]))))

    def _repr_html_(self):
        """For use with IPython.display.display()
        
        TODO: all the information of interest is in self._stacks. Quite how much
        effort you want to go to in order to display it nicely is up to you!
        
        The version below seems to be reasonable for now.
        
        We aggregate on stack_list and type. Then, if the messages differ for
        that {stack_list and type} we sub aggregate by message string.        
        We make no effort to distinguish between warnings and exceptions.
        """
        
        aggregates = {}
        for state,message,x_type,stack_list in self._stacks:
            message = str(message)
            k = (x_type,stack_list)
            
            # if k already exists as a key in aggregates then use that instead, 
            # note the two things are not "is" true, they are only "==" true.
            try:
                k = (a for a in aggregates if a == k).next() 
            except Exception:
                aggregates[k] = [] # it didn't exist so make it exist
                pass           
            
            aggregates[k].append((state,message))
                

        s = []
        for (x_type,stack_list), state_message_list in aggregates.iteritems():

            if all((m==state_message_list[0][1] for _,m in state_message_list)): # all messages are the same for this particular trace
                state_list = map(lambda x: x[0],state_message_list)
                msg_str =  '"%s" { BR }%d occurrences: { SPAN_A }%s{/ SPAN }{ BR }' % (state_message_list[0][1],len(state_list), ', '.join(state_list) )
            else:
                u = []
                for message, sub_state_message_list in groupby(sorted(state_message_list,key=lambda x:x[1]),lambda x:x[1]):
                    state_list = map(lambda x: x[0],sub_state_message_list)
                    u.append('"%s"{ BR }%d occurrences: { SPAN_A }%s{/ SPAN }' % (message,len(state_list),', '.join(state_list)))
                msg_str = '%d total occurrences with %d unique messages:{ BR }%s{ BR }' % (len(state_message_list),len(u),'{ BR }'.join(u))
                
            s.append(format_list2(x_type.__name__ if not isinstance(x_type,str) else x_type, msg_str, stack_list))
            
        n_unique = len(s)
        s = '<BR>'.join(s).replace('{ SPAN_A }','<span style="color:rgb(150,150,150);">')\
                          .replace('{ SPAN_B }','<span style="color:rgb(200,30,30);">')\
                          .replace('{ SPAN_C }','<span style="color:rgb(100,100,255);">')\
                          .replace('{/ SPAN }',"</span>").replace('{ BR }','<BR>')
        return'<BR><B><span style="text-decoration: underline;">AGGREGATE EXCEPTIONS LOG</span></B><BR><BR>'+ s + \
        '<BR><span style="text-decoration: underline;">Summary of warnings/exceptions</span>:<BR><B>Total:</B> %d<BR><B>Unique: </B>%d' %(len(self._stacks),n_unique) 

    def __str__(self):
        """ Strips out the formatting for the html representation.
            So that it can be rendered more simply.
            
            Regex from http://stackoverflow.com/a/4869782/2399799
        """
        html = self._repr_html_()
        return re.sub('<[^<]+?>', '', html.replace('<BR>','\n')).replace('&lt;','<') .replace('&gt;','>').replace('&nbsp;',' ')

    def __enter__(self):
        return self
    
    def __exit__(self,type, value, tb ):
        if len(self._stacks):
            display(self)
            
            
class _StateCM(object):
    """
    Do not create instances directly, instead, see ExceptionAggregator.state method.    
    This should only be used in a with-block..it doesn't make sense in any other form.
    """
    def __init__(self,parentXA,state):
        self._parentXA = parentXA
        self._state = state
        
    def __enter__(self):
        self._ctx_stack = extract_stack()[:-2] # store this, so we know what to trim off when logging warnings
        self._old_showwarning = warnings.showwarning #we store this so we can restore it
        warnings.showwarning = self._logwarning
        
    def __exit__(self,type, value, tb ):
        warnings.showwarning = self._old_showwarning #restore this
        if type is not None:
            if type is KeyboardInterrupt:
                raise KeyboardInterrupt
            self._parentXA.trial_log(self._state) # an exception occured
        return True #silently swallow exceptions
            
    def _logwarning(self, message, category, filename, lineno, file=None):
        """
        This function recieves the warning events rather than the default write-to-stdout
        function that warnings usually has.
        
        """
        sys.stdout.flush()
        print(self._parentXA.warning_symbol,end='')
        sys.stdout.flush()
        
        nice_stack = extract_stack()[len(self._ctx_stack):-1] #trim off top of stack above with-block (or leave fully intact if called not using "with")
        self._parentXA._stacks.append((self._state,message,category.__name__,tuple(nice_stack)))
        
        

@contextmanager
def ignored(*exceptions):
    try:
        yield
    except exceptions:
        pass