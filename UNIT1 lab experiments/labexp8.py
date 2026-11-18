p_m1 = 0.60
p_m2 = 0.40
defect_m1 = 0.02
defect_m2 = 0.05

p_defect = p_m1 * defect_m1 + p_m2 * defect_m2
p_m1_given_defect = p_m1 * defect_m1 / p_defect

print("Probability of a defective product:", p_defect)
print("Probability defective product came from Machine 1:", p_m1_given_defect)
