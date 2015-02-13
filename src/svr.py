from __future__ import print_function

###############################################################################
# Imports
import numpy as npy
#import scipy
import pylab as pl

from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

from sklearn.cross_validation import train_test_split

###############################################################################
# Data

mUrl = npy.array([3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 64, 65, 66]).astype(float)
tUrl = npy.array([-4, 11, 17, 18, 16, 10, 3, -5, -12, -15, -12, -4, 5, 11, 17, 18, 16, 10, -15, -12, -4, 5, 11, 17, 18, 16, 10, 3, -5, -12, -15, -12, -4, 5, 11, 17, 18, 16, 10, 3, -5, -12, -15, -12, -4, 5, 11, 17, 18, 16, 3, -5, -12, -15, -12, -4, 5, 11, 17]).astype(float)

mUrlForecast = npy.array([3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 73, 74, 75]).astype(float)
tUrlForecast = npy.array([-4, 11, 17, 18, 16, 10, 3, -5, -12, -15, -12, -4, 5, 11, 17, 18, 16, 10, -15, -12, -4, 5, 11, 17, 18, 16, 10, 3, -5, -12, -15, -12, -4, 5, 11, 17, 18, 16, 10, 3, -5, -12, -15, -12, -4, 5, 11, 17, 18, 16, 3, -5, -12, -15, -12, -4, 5, 11, 17, 16, 10, 3, -5, -12, -15, -12, -4]).astype(float)

mFzl = npy.array([1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]).astype(float)
tFzl = npy.array([-15, -12, -4, 5, 17, 18, 16, 3, -5, -12, -15, -12, -4, 5, 11, 17, 18, 16, -15, -12, -4, 5, 11, 17, 18, 16, 10, 3, -5, -12, -15, -12, -4, 5, 11, 17, 18, 16, 10, 3, -5, -12]).astype(float)

mFzlForecast = npy.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75]).astype(float)
tFzlForecast = npy.array([-15, -12, -4, 11, 5, 17, 18, 16, 3, -5, -12, -15, -12, -4, 5, 11, 17, 18, 16, -15, -12, -4, 5, 11, 17, 18, 16, 10, 3, -5, -12, -15, -12, -4, 5, 11, 17, 18, 16, 10, 3, -5, -12, -15, -12, 5, 11, 17, 18, 16, 10, 3, -5, -12, -15, -12, -4, 5, 11, 17, 18, 16, 10, 3, -5, -12, -15, -12, -4]).astype(float)

kwhUrl = npy.array([2303201, 1899025, 2284937, 1815803, 1996193, 2309336, 2811586, 2433297, 2972759, 2831664, 2705616, 2086305, 1876835, 1876835, 1721392, 1607396, 1680200, 1718125, 2459661, 2442159, 1961160, 3592994, 2825545, 2056561, 1988015, 2887907, 2865541, 4119753, 4839448, 5288796, 5170509, 4457033, 4613986, 4214131, 3361703, 2767700, 3225771, 3723320, 5052361, 5796639, 6733936, 6976714, 7138770, 7205501, 6229614, 5934359, 4253119, 3884593, 3865992, 4180972, 6129750, 6769443, 7679900, 7154115, 9777309, 9238004, 9190574, 7965749, 7415240]).astype(float)
kwhFiz = npy.array([3794692, 3386642, 3027843, 2998446, 2931874, 3128901, 3196637, 3438611, 3424399, 3560712, 4087491, 4089454, 3962090, 2567271, 2567271, 3723664, 3268302, 3486375, 4879726, 4555684, 4371151, 3354729, 3436386, 3445132, 3352583, 2723183, 2930647, 3219781, 3584716, 3860956, 3981839, 3793177, 4209961, 3272643, 3381616, 3085526, 2867567, 3088562, 3202499, 3269441, 3693039, 3409947]).astype(float)

###############################################################################
# Scalers

m_scaler = preprocessing.StandardScaler().fit(mUrl)
t_scaler = preprocessing.StandardScaler().fit(tUrl)
kwhUrl_scaler = preprocessing.StandardScaler().fit(kwhUrl)
kwhFiz_scaler = preprocessing.StandardScaler().fit(kwhFiz)

###############################################################################
# Scaled values

mtUrlForecast_scaled = npy.array(zip(m_scaler.transform(mUrlForecast), t_scaler.transform(tUrlForecast)))
mtFzlForecast_scaled = npy.array(zip(m_scaler.transform(mFzlForecast), t_scaler.transform(tFzlForecast)))

tUrl_scaled = t_scaler.transform(tUrl)
tFzl_scaled = t_scaler.transform(tFzl)

mtUrl_scaled = npy.array(zip(m_scaler.transform(mUrl), t_scaler.transform(tUrl)))
mtFzl_scaled = npy.array(zip(m_scaler.transform(mFzl), t_scaler.transform(tFzl)))

#mtUrl_scaled = npy.array(m_scaler.transform(mUrl) + t_scaler.transform(tUrl))
#mtFzl_scaled = npy.array(m_scaler.transform(mFzl) + t_scaler.transform(tFzl))

kwhUrl_scaled = kwhUrl_scaler.transform(kwhUrl)
kwhFiz_scaled = kwhFiz_scaler.transform(kwhFiz)

print(npy.size(mtUrl_scaled))
print (npy.size(kwhUrl_scaled))

###############################################################################
# Get a test set
mtUrl_train, mtUrl_test, kwhUrl_train, kwhUrl_test = train_test_split(mtUrl_scaled, kwhUrl_scaled, test_size = 0.1, random_state = 0)
mtFzl_train, mtFzl_test, kwhFzl_train, kwhFzl_test = train_test_split(mtFzl_scaled, kwhFiz_scaled, test_size = 0.1, random_state = 0)
###############################################################################
# Fit regression model

svrUrl_rbf = SVR(kernel = 'rbf', C = 1e5, gamma = 0.003)
svrUrl_lin = SVR(kernel = 'linear', C = 1e3)
svrUrl_poly = SVR(kernel = 'poly', C = 1e4, degree = 2)
kwhUrlPredictRBF_clf = svrUrl_rbf.fit(mtUrl_scaled, kwhUrl_scaled)
kwhUrlPredictRBF_scaled = kwhUrlPredictRBF_clf.predict(mtFzlForecast_scaled)
kwhUrlPredictLin_scaled = svrUrl_lin.fit(mtUrl_scaled, kwhUrl_scaled).predict(mtFzlForecast_scaled)
kwhUrlPredictPoly_scaled = svrUrl_poly.fit(mtUrl_scaled, kwhUrl_scaled).predict(mtFzlForecast_scaled)

pl.scatter(mUrl, kwhUrl_scaler.inverse_transform(kwhUrl_scaled), c = 'r', label = 'KWh')
pl.hold('on')
for c in [1e1, 1e2, 1e3, 1e4, 1e5]:
	for g in [0.1, 0.03, 0.01, 0.003]:
		svrUrl_rbf_cv = SVR(kernel = 'rbf', C = c, gamma = g)
		kwhUrlPredictRBF_clf_cv = svrUrl_rbf_cv.fit(mtUrl_scaled, kwhUrl_scaled)
		print(kwhUrlPredictRBF_clf_cv.score(mtUrl_test, kwhUrl_test))
		kwhUrlPredictRBF_scaled_cv = kwhUrlPredictRBF_clf_cv.predict(mtFzlForecast_scaled)

		pl.plot(mFzlForecast, kwhUrl_scaler.inverse_transform(kwhUrlPredictRBF_scaled_cv), c = 'g', label = 'c ' + str(c) + ' g ' + str(g))
#pl.legend(loc = 'upper left')
pl.show()


print('cv error:')
print(kwhUrlPredictRBF_clf.score(mtUrl_test, kwhUrl_test))

svrFiz_rbf = SVR(kernel = 'rbf', C = 1e6, gamma = 0.003)
svrFiz_lin = SVR(kernel = 'linear', C = 1e3)
svrFiz_poly = SVR(kernel = 'poly', C = 1e4, degree = 2)
kwhFizPredictRBF_scaled = svrFiz_rbf.fit(mtFzl_scaled, kwhFiz_scaled).predict(mtFzlForecast_scaled)
kwhFizPredictLin_scaled = svrFiz_lin.fit(mtFzl_scaled, kwhFiz_scaled).predict(mtFzlForecast_scaled)
kwhFizPredictPoly_scaled = svrFiz_poly.fit(mtFzl_scaled, kwhFiz_scaled).predict(mtFzlForecast_scaled)

y_rbf_fizl_sum = svrFiz_rbf.fit(mtFzl_scaled, kwhFiz_scaled).predict(mtFzlForecast_scaled)
y_lin_fizl_sum = svrFiz_lin.fit(mtFzl_scaled, kwhFiz_scaled).predict(mtFzlForecast_scaled)

y_sum = npy.add(kwhFiz_scaler.inverse_transform(y_lin_fizl_sum), kwhUrl_scaler.inverse_transform(kwhUrlPredictRBF_scaled))

###############################################################################
# look at the results
pl.scatter(mUrl, kwhUrl_scaler.inverse_transform(kwhUrl_scaled), c = 'r', label = 'data ur.l')
#pl.scatter(mUrl, tUrl_scaled, c='r', label='temp', marker='+')
pl.scatter(mFzl, kwhFiz_scaler.inverse_transform(kwhFiz_scaled), c = 'y', label = 'data fiz.l')
#pl.scatter(mFzl, tFzl_scaled, c='b', label='temp', marker='+')

pl.hold('on')
pl.plot(mFzlForecast, kwhUrl_scaler.inverse_transform(kwhUrlPredictRBF_scaled), c = 'g', label = 'RBF model ur.l')
pl.plot(mFzlForecast, kwhFiz_scaler.inverse_transform(kwhFizPredictLin_scaled), c = 'y', label = 'model fiz.l')
pl.plot(mFzlForecast, y_sum, c = 'k', label = 'model sum')
pl.plot(mFzlForecast, kwhUrl_scaler.inverse_transform(kwhUrlPredictLin_scaled), c = 'r', label = 'Linear model')
pl.plot(mFzlForecast, kwhUrl_scaler.inverse_transform(kwhUrlPredictPoly_scaled), c = 'b', label = 'Polynomial model')
pl.xlabel('months')
pl.ylabel('target')
pl.title('Support Vector Regression')
pl.legend(loc = 'upper left')
pl.show()
