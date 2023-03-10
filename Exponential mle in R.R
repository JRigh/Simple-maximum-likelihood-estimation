#-------------------------------
# maximum likelihood estimation
# exponential model (07.01.2023)
#-------------------------------

# generate a random sample of n = 5000 from an Exponential distribution
# with parameter lambda = 1.75 (argument name = 'rate')
set.seed(1986)
n = 5000
xi <- rexp(n = n, rate = 1.75) # the rate is supposed to be unknown
head(xi)
# [1] 0.4817649 0.5847904 0.1148642 0.1820848 0.9921011 0.2747533

# Closed-form MLE
lambda_hat_formula = n / sum(xi)   # or 1 / mean(xi)
lambda_hat_formula # [1] 1.744645

# Numerical approximation of the MLE 
mle = optimize(function(lambda){sum(dexp(x = xi, rate = lambda, log = TRUE))},
               interval = c(0, 10),
               maximum = TRUE,
               tol = .Machine$double.eps^0.5)

lambda_hat = mle$maximum
lambda_hat # [1] 1.744645

# Plot the Log-Likelihood function
possible.lambda <- seq(0, 10, by = 0.0001)

qplot(possible.lambda,
      sapply(possible.lambda, function (lambda) {sum(dexp(x = xi, rate = lambda, log = TRUE))}),
      geom = 'line',
      xlab = 'lambda',
      ylab = 'Log-Likelihood') +
  geom_vline(xintercept = lambda_hat, color = 'red', size=1.1) +
  geom_line(size = 1.1) + 
  labs(title = 'Log-Likelihood (function of lambda)',
       subtitle = "Maximum is reached at lambda = 1.744645",
       caption = "Artificial dataset of size 5000") +
  theme(axis.text = element_text(size = 8),
        axis.title = element_text(size = 10),
        plot.subtitle = element_text(size = 9, face="italic", color="darkred"),
        panel.background = element_rect(fill = "white", colour = "grey50"),
        panel.grid.major = element_line(colour = "grey90"))

#----
# end
#----
