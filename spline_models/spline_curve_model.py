"""
@author: Virginie UHLMANN

Please acknowledge as Uhlmann group's Spline Fitting Toolbox v1.0
"""

import numpy as np

class B3_spline_curve:
    wrongDimensionMessage = 'It looks like coefs is a 2D array with second dimension different than two. I don\'t know how to handle this.'
    wrongArraySizeMessage = 'It looks like coefs is not a 2D array. I don\'t know how to handle this.'
    noCoefsMessage = 'This model doesn\'t have any coefficients.'
    unimplementedMessage = 'This function is not implemented.'

    def __init__(self, M, closed):
        self.gen = B3()

        if M >= self.gen.support:
            self.M = M
        else:
            raise RuntimeError('M must be greater or equal than 4.')
            return

        self.closed = closed
        self.coefs = None
        self.halfSupport = self.gen.support/2.0

    def sample(self, samplingRate):
        if self.coefs is None:
            raise RuntimeError(self.noCoefsMessage)
            return

        if len(self.coefs.shape) == 2 and self.coefs.shape[1] == 2:
            if self.closed:
                N = samplingRate * self.M
            else:
                N = (samplingRate * (self.M - 1)) + 1
            
            curve = [self.parameterToWorld(float(i) / float(samplingRate)) for i in range(0, N)]
        else:
            raise RuntimeError(self.wrongArraySizeMessage)
            return

        return np.stack(curve)

    def get_spline_coefficients(self, knots):
        knots = np.array(knots)
        if len(knots.shape) == 2:
            if (knots.shape[1] == 2):
                if self.closed:
                    coefsX = self.gen.filterPeriodic(knots[:, 0])
                    coefsY = self.gen.filterPeriodic(knots[:, 1])
                else:
                    knots=np.vstack((knots,knots[-1]))
                    knots=np.vstack((knots,knots[-1]))
                    knots=np.vstack((knots[0],knots))
                    knots=np.vstack((knots[0],knots))
                    coefsX = self.gen.filterSymmetric(knots[:, 0])
                    coefsY = self.gen.filterSymmetric(knots[:, 1])
                self.coefs = np.hstack((np.array([coefsX]).transpose(), np.array([coefsY]).transpose()))
            else:
                raise RuntimeError(self.wrongDimensionMessage)
                return
        else:
            raise RuntimeError(self.wrongArraySizeMessage)
            return

        return

    def parameterToWorld(self, t, dt=False):
        if self.coefs is None:
            raise RuntimeError(SplineCurve.noCoefsMessage)
            return

        value = 0.0
        if self.closed:
            for k in range(0, self.M):
                tval = self.wrapIndex(t, k)
                if (tval > -self.halfSupport and tval < self.halfSupport):
                    if dt:
                        splineValue=self.gen.firstDerivativeValue(tval)
                    else:
                        splineValue=self.gen.value(tval)
                    value += self.coefs[k] * splineValue

        else:
            for k in range(0, self.M+int(self.gen.support)):
                tval = t - (k - self.halfSupport)
                if (tval > -self.halfSupport and tval < self.halfSupport):
                    if dt:
                        splineValue=self.gen.firstDerivativeValue(tval)
                    else:
                        splineValue=self.gen.value(tval)
                    value += self.coefs[k] * splineValue
        return value

    def wrapIndex(self, t, k):
        wrappedT = t - k
        if k < t - self.halfSupport:
            if k + self.M >= t - self.halfSupport and k + self.M <= t + self.halfSupport:
                wrappedT = t - (k + self.M)
        elif k > t + self.halfSupport:
            if k - self.M >= t - self.halfSupport and k - self.M <= t + self.halfSupport:
                wrappedT = t - (k - self.M)
        return wrappedT

    def centroid(self):
        centroid=np.zeros((2))
        
        for k in range(0, self.M):
            centroid+=self.coefs[k]

        return centroid/self.M

    def translate(self, translationVector):
        for k in range(0, self.M):
            self.coefs[k]+=translationVector

    def scale(self, scalingFactor):
        centroid=self.centroid()
        
        for k in range(0, self.M):
            vectorToCentroid=self.coefs[k]-centroid
            self.coefs[k]=centroid+scalingFactor*vectorToCentroid

    def rotate(self, rotationMatrix):
        for k in range(0, self.M):
            self.coefs[k]=np.matmul(rotationMatrix, self.coefs[k])

class B3():
    def __init__(self):
        self.support=4.0

    def value(self, x):
        val = 0.0
        if 0 <= abs(x) and abs(x) < 1:
            val = 2.0 / 3.0 - (abs(x) ** 2) + (abs(x) ** 3) / 2.0
        elif 1 <= abs(x) and abs(x) <= 2:
            val = ((2.0 - abs(x)) ** 3) / 6.0
        return val

    def firstDerivativeValue(self, x):
        val = 0.0
        if 0 <= x and x < 1:
            val = -2.0 * x + 1.5 * x * x
        elif -1 < x and x < 0:
            val = -2.0 * x - 1.5 * x * x
        elif 1 <= x and x <= 2:
            val = -0.5 * ((2.0 - x) ** 2)
        elif -2 <= x and x <= -1:
            val = 0.5 * ((2.0 + x) ** 2)
        return val

    def secondDerivativeValue(self, x):
        val = 0.0
        if 0 <= x and x < 1:
            val = -2.0 + 3.0 * x
        elif -1 < x and x < 0:
            val = -2.0 - 3.0 * x
        elif 1 <= x and x <= 2:
            val = (2.0 - x)
        elif -2 <= x and x <= -1:
            val = (2.0 + x)
        return val

    def filterSymmetric(self, s):
        M = len(s)
        pole = -2.0 + np.sqrt(3.0)

        cp = np.zeros(M)
        eps = 1e-8
        k0 = np.min(((2 * M) - 2, int(np.ceil(np.log(eps) / np.log(np.abs(pole))))))
        for k in range(0, k0):
            k = k % (2 * M - 2)
            if k >= M:
                val = s[2 * M - 2 - k]
            else:
                val = s[k]
            cp[0] += (val * (pole ** k))
        cp[0] *= (1.0 / (1.0 - (pole ** (2 * M - 2))))

        for k in range(1, M):
            cp[k] = s[k] + pole * cp[k - 1]

        cm = np.zeros(M)
        cm[M - 1] = cp[M - 1] + (pole * cp[M - 2])
        cm[M - 1] *= (pole / ((pole ** 2) - 1))
        for k in range(M - 2, -1, -1):
            cm[k] = pole * (cm[k + 1] - cp[k])

        c = cm * 6.0

        c[np.where(abs(c) < eps)] = 0.0
        return c

    def filterPeriodic(self, s):
        M = len(s)
        pole = -2.0 + np.sqrt(3.0)

        cp = np.zeros(M)
        for k in range(0, M):
            cp[0] += (s[(M - k) % M] * (pole ** k))
        cp[0] *= (1.0 / (1.0 - (pole ** M)))

        for k in range(1, M):
            cp[k] = s[k] + pole * cp[k - 1]

        cm = np.zeros(M)
        for k in range(0, M):
            cm[M - 1] += ((pole ** k) * cp[k])
        cm[M - 1] *= (pole / (1.0 - (pole ** M)))
        cm[M - 1] += cp[M - 1]
        cm[M - 1] *= (-pole)

        for k in range(M - 2, -1, -1):
            cm[k] = pole * (cm[k + 1] - cp[k])

        c = cm * 6.0

        eps = 1e-8
        c[np.where(abs(c) < eps)] = 0.0
        return c

