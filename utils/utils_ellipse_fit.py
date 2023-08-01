# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt

class Ellipse(object):

    def __init__(self):
        """
        All* the actual maths seems to be from:
            http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
        although I got it via Robin.
        
        I turned it into a class.
        
        Use ``Ellipse.from_xy(x,y)`` rather than creating directly.
        
        * the transformation matrix is from elsewhere (see method for details)
        """
        self._isok = False
    
    @staticmethod
    def from_xy(x, y):
        x = x[:,np.newaxis]
        y = y[:,np.newaxis]
        
        ret = Ellipse()
        ret._x = x
        ret._y = y

        D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
        S = np.dot(D.T,D)
        C = np.zeros([6,6])
        C[0,2] = C[2,0] = 2; C[1,1] = -1
        try:
            E, V =  np.linalg.eig(np.dot(np.linalg.inv(S), C))
        except np.linalg.LinAlgError:
            return ret
        n = np.argmax(np.abs(E))
        a = V[:,n]
        ret._a = a
        ret._isok = True
        return ret

    @property
    def centre(self):
        if not hasattr(self, '_centre'):
            a = self._a
            b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
            num = b*b-a*c
            x0=(c*d-b*f)/num
            y0=(a*f-b*d)/num
            self._centre = np.array([x0,y0])
        return self._centre
        
    @property
    def angle(self):
        if not hasattr(self, '_angle'):
            a = self._a
            b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
            self._angle = 0.5*np.arctan(2*b/(a-c))
        return self._angle
    
    @property
    def axis_length(self):
        if not hasattr(self, '_axis_length'):
            a = self._a
            b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
            up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
            down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
            down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
            res1=np.sqrt(up/down1)
            res2=np.sqrt(up/down2)
            self._axis_length = np.array([res1, res2])
        return self._axis_length

    def is_ok(self):
        """Returns False if the angle, centre, or axes don't seem good, or
        if the ellipse was not properly initalised."""
        if not self._isok:
            return False
        return not (np.any(np.isnan(self.centre))  or
                    np.any(np.isnan(self.axis_length)) or
                    np.isnan(self.angle))
            
    @property
    def to_circle_transform_matrix(self):
        """computes the matrix needed to transform the ellipse into a circle.

        Translate centre to origin, rotate to align axes with cartesian axes,
        rescale each axis, then rotate back, and traslate back.
        
        Use as ``np.dot(mat, xy1)``, where xy1 is a 3xn array, with ones in the
        third row.
        """
        if not hasattr(self, '_transmat'):
            (a, b), ang, centre = self.axis_length, self.angle, self.centre
            C = np.matrix([[1, 0, -centre[0]],
                           [0, 1, -centre[1]],
                           [0, 0 , 1]])
            C_ = np.matrix([[1, 0, centre[0]],
                            [0, 1, centre[1]],
                            [0 , 0, 1]])
            R = np.matrix([[np.cos(ang), np.sin(ang), 0],
                            [-np.sin(ang), np.cos(ang), 0],
                             [0, 0, 1]])
            S = np.matrix([[np.mean([a,b])/a, 0, 0],
                            [0, np.mean([a,b])/b, 0],
                             [0, 0, 1]])
            self._transmat = C_ * R.T * S * R * C 
            
        return self._transmat
        
    @property
    def from_circle_transform_matrix(self):
        """computes the matrix needed to transform the a circle into the ellipse.
        For a given affine transform matrix, the inverse transform is not simply
        the inverse of the matrix, you have to do something a little more fiddly.
        Taken from wikipedia.
        """
        M = self.to_circle_transform_matrix
        A_ = np.linalg.inv(M[0:2,0:2])
        b = M[0:2,2]
        return np.vstack((np.hstack((A_,-np.dot(A_ ,b))),[0,0,1]))
        
    def get_perim_XY(self, pts=100):
        '''returns ``pts=100`` x and y points for plotting of the ellipse
        '''
        (a, b), ang, centre = self.axis_length, self.angle, self.centre
        
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        theta = np.linspace(0, 2*np.pi, pts)
        X = a*np.cos(theta)*cos_a - sin_a*b*np.sin(theta) + centre[0]
        Y = a*np.cos(theta)*sin_a + cos_a*b*np.sin(theta) + centre[1]
        return np.array((X,Y))
    
    def plot(self, flip=False):
        x = getattr(self,'_x',[])
        y = getattr(self,'_y',[])
        centre = self.centre
        p = self.get_perim_XY()

        if flip:
            x, y = y, x
            centre = centre[::-1]
            p = p[::-1]
            
        plt.plot(x.ravel(), y.ravel(),'.r')
        plt.plot(centre[0], centre[1], 'bx')
        plt.plot(p[0],p[1],'b-')
        plt.axis('equal')

    
    
