prevalence = 0.01
sensitivity = 0.95
specificity = 0.90

p_positive = sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
p_disease_given_positive = sensitivity * prevalence / p_positive

print("Probability of positive test:", p_positive)
print("Probability disease given positive:", p_disease_given_positive)
