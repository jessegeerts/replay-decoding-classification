# -*- coding: utf-8 -*-

import inspect
import matplotlib.pylab as plt
from utils.custom_mpl import figure_to_html
import subprocess

class TrialSummary(object):
    """ a mixin for trial, adding some rough methods for summarising the 
        outputs of multiple analysis methods."""
        
    def make_all_plots(self, t, c, fn="summary_plots.html", plot_names=None):
        """ Just a bit of fun...tries running all the plotting functions for 
        the given cell, opens each in a new figure if fn is None, or
        saves to html page if fn is a string filename.
        
        If plot_names is a sequence of strings only those functions will be
        called, otherwise all plot* functions will be called.
        """
        if fn is None:
            as_html = False
            plt.rcParams['figure.max_open_warning'] = 100
        else:
            as_html = True
            plt.ioff()
            if not fn.endswith(".html"):
                raise ValueError("fn must be a filename ending in '.html'")
        orig_nfig_warning = plt.rcParams['figure.max_open_warning']
        
        html = ["<html><title>{} t{}c{}</title>".format(self.experiment_name, t, c),
                "<style>.plot{box-shadow: 0 4px 5px 0 rgba(0, 0, 0, 0.14),",
                              "0 1px 10px 0 rgba(0, 0, 0, 0.12), 0 2px 4px ",
                              "-1px rgba(0, 0, 0, 0.4);display:inline-block;",
                              "margin:10px;}</style>",
                "<body style='text-align:center'>",
                "<h2>Trial '{}'</h2>".format(self.experiment_name)]
        if plot_names is None:
            plot_names = (foo for foo in dir(self) if foo.startswith('plot'))
            
        for ii, foo_name in enumerate(plot_names):
            if as_html:
                plt.clf()
            else:
                plt.figure(ii+1)
            # plt.hold(True) # clf does not reset hold to True             
            try:
                foo = getattr(self, foo_name)
                foo_args = inspect.getargspec(foo).args
                if foo_args[1:3] == ['t', 'c']:
                    foo(t, c)
                    print("figure(%d): %s" % (ii+1, foo_name))
                    title = "{}(t={}, c={})".format(foo_name, t, c)
                else:
                    foo()
                    print("figure(%d): %s [ignored t and c]" % (ii+1, foo_name))
                    title = "{}()".format(foo_name)
                if as_html:
                    html.append( "<div class='plot'><h3>{}</h3>{}</div>".format(title, figure_to_html()))
                    plt.close()
                else:
                    plt.gcf().canvas.set_window_title("figure(%d): %s" % (ii+1, foo_name))
            except Exception as ex:
                print("Failed to run ({}) {}:\n\t{}: {}".format(ii+1, foo_name,
                                  type(ex).__name__, str(ex)))              
                plt.close()
        if as_html:
            plt.ion()
            html.append("</body></html>")
            with open(fn, "w") as f:
                f.write(''.join(html))
            subprocess.Popen([fn], shell=True)
        else:
            plt.rcParams['figure.max_open_warning'] = orig_nfig_warning

