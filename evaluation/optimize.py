import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# ==> Create a solver and load the model
opt_solver = SolverFactory('mindtpy')
model = pyo.ConcreteModel()

# ==> Define the variables
# 1. number_chunks_limit: int
bound_N_s = (3, 300)
model.N_s = pyo.Var(within=pyo.NonNegativeIntegers, bounds=bound_N_s)
# 2. number_chunks_rerank: int
bound_N_r = (1, 20)
model.N_r = pyo.Var(within=pyo.NonNegativeIntegers, bounds=bound_N_r)
# 3. hybrid_search_alpha: float
model.alpha = pyo.Var(within=pyo.UnitInterval, bounds=(0, 1))
# 4. temperature_LLM: float
model.T = pyo.Var(within=pyo.UnitInterval, bounds=(0, 1))
# 5. penalty_frequency_LLM: float
model.P_f = pyo.Var(bounds=(0, 2))
# 6. if_multi_step_RAG: bool
model.MSR = pyo.Var(within=pyo.Binary)
# 7. if_contextual_embedding: bool
model.CE = pyo.Var(within=pyo.Binary)


# ==> Generate random initial points
import random

def generate_random_initial_points():
    N_s_val = random.randint(bound_N_s[0], bound_N_s[1])
    N_r_val = random.randint(bound_N_r[0], bound_N_r[1])
    alpha_val = random.uniform(0, 1)
    T_val = random.uniform(0, 1)
    P_f_val = random.uniform(0, 2)
    MSR_val = random.choice([0, 1])
    CE_val = random.choice([0, 1])
    return {'N_s': N_s_val, 'N_r': N_r_val, 'alpha': alpha_val, 'T': T_val, 'P_f': P_f_val, 'MSR': MSR_val, 'CE': CE_val}

random_init=generate_random_initial_points()
model.N_s.set_value(random_init['N_s'])
model.N_r.set_value(random_init['N_r'])
model.alpha.set_value(random_init['alpha'])
model.T.set_value(random_init['T'])
model.P_f.set_value(random_init['P_f'])
model.MSR.set_value(random_init['MSR'])
model.CE.set_value(random_init['CE'])

print("Initial model:")
model.display()


# ==> Define the objective function
from evaluation.services.evaluation_steps import EvaluationSteps
from evaluation.calculate_loss import calculate_loss
eva = EvaluationSteps(
    number_chunks_limit=model.N_s.value,
    hybrid_search_alpha=model.alpha.value,
    number_chunks_rerank=model.N_r.value,
    temperature_LLM=model.T.value,
    penalty_frequency_LLM = model.P_f.value,
    if_multi_step_RAG= model.MSR.value,
    if_contextual_embedding = model.CE.value
)

model.loss = pyo.Objective(expr=calculate_loss(eva), sense=pyo.minimize)

results = opt_solver.solve(model,strategy='GOA', tee=True,use_mcpp = True, add_cuts_at_incumbent = True, time_limit=3600)

eva.context_provider.close()

# ==> Display the results
print(f"Solver status: {results.solver.status}")
print(f"Termination condition: {results.solver.termination_condition}")
print(f"Optimal N_s: {model.N_s()}")
print(f"Optimal N_r: {model.N_r()}")
print(f"Optimal alpha: {model.alpha()}")
print(f"Optimal T: {model.T()}")
print(f"Optimal P_f: {model.P_f()}")
print(f"Optimal MSR: {model.MSR}")
print(f"Optimal CE: {model.CE}")
print(f"Minimum loss: {model.loss()}")