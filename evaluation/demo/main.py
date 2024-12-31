from pyomo.environ import *
from pyomo.contrib.mindtpy.MindtPy import MindtPySolver


def accuracy_calculation(N_s, N_r, T, P_f, M, C):
    return 0.5 * N_s + 0.3 * N_r + 0.1 * T + 0.05 * P_f + 0.05 * M + 0.05 * C


def time_calculation(N_s, N_r, T, P_f, M, C):
    return 0.1 * N_s + 0.2 * N_r + 0.3 * T + 0.2 * P_f + 0.1 * M + 0.1 * C

# SolverFactoryClass('mindtpy').solve(model, mip_solver='glpk', nlp_solver='ipopt') 


model = ConcreteModel()

model.N_s = Var(domain=Integers, bounds=(3, 300))   # Number of chunks retrieved
model.N_r = Var(domain=Integers, bounds=(1, 10))    # Number of chunks picked
model.T = Var(domain=Reals, bounds=(0.0, 1.0))      # LLM temperature
model.P_f = Var(domain=Reals, bounds=(-2.0, 2.0))   # Frequency penalty
model.M = Var(domain=Binary)                        # Multi-step
model.C = Var(domain=Binary)                        # Contextual embedding


# Stage 1 - Minimize loss
def loss_rule(m):
    # Replace with your black-box function or an approximation
    return 1 - accuracy_calculation(m.N_s, m.N_r, m.T, m.P_f, m.M, m.C)

model.loss = Objective(rule=loss_rule, sense=minimize)

# Stage 2 - Trade-off between loss and time

model.time = Expression(expr=time_calculation(model.N_s, model.N_r, model.T, model.P_f, model.M, model.C))


model.stage2 = Objective(expr=0.7 * model.loss + 0.3 * model.time, sense=minimize)

solver = MindtPySolver()
solver.solve(model, strategy='OA') 

print(f"Optimal N_s: {model.N_s()}")
print(f"Optimal N_r: {model.N_r()}")
print(f"Optimal T: {model.T()}")
print(f"Optimal P_f: {model.P_f()}")
print(f"Optimal M: {model.M()}")
print(f"Optimal C: {model.C()}")
