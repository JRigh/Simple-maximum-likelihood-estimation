#-------------------------------
# maximum likelihood estimation
# exponential model (07.01.2023)
#-------------------------------

from scipy import stats
import numpy as np

# generate a sample of size 5000
# in numpy, the parametrization is different from R
np.random.seed(1986)
n = 5000
Lambda = 1.75 # true value of the parameter, that we wish to estimate
xi = np.random.exponential(scale = 1/Lambda, size = n)
print(xi)
# [0.48077598 0.04599578 0.53583846 ... 0.31828503 0.05858069 0.0721613 ]

# Closed-form MLE
lambda_hat_formula = n / sum(xi)   # or 1 / statistics.mean(xi)
lambda_hat_formula 
# [1] 1.71848223876711

def llikelihood(Lambda):  
    # log-likelihood function
    ll = -np.sum(stats.expon.logpdf(xi, scale = 1/Lambda))
    return ll

from scipy.optimize import minimize

# Numerical approximation of the MLE using minimize()
mle = minimize(llikelihood, 
                   x0 = 4, 
                   method = 'BFGS')
print(mle.x)
# [1.71848223]

# Plot the Log-Likelihood function
import matplotlib.pyplot as plt

# set a range of possible parameter values
possiblelambda = np.linspace(start = 0, stop = 10, num = 5000)

# compute the log-likelihood function for all possible parameter values
LL = []
for L in possiblelambda:
    loglikelihood = np.log(L)- L * xi
    LL.append(loglikelihood.sum())

# plotting using matplotlib
plt.plot(possiblelambda, LL, color = 'black')
plt.axvline(x = mle.x,  color = 'red')
plt.xlabel('lambda')
plt.ylabel('Log-Likelihood')
plt.title('Log-Likelihood (function of lambda)', fontsize = 15, y = 1.1, loc = 'left')
plt.suptitle('Maximum is reached at lambda = 1.71848223', y = 0.95, color = 'darkred', x = 0.45)
caption = 'artificial dataset of size 5000'
plt.text(6.5,-38000, caption, fontsize = 8)
plt.grid(color = 'whitesmoke', linestyle = '-', linewidth = 1.2)
plt.show()

#----
# end
#----
