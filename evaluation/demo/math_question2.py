from pyomo.environ import *
from pyomo.contrib.mindtpy.MindtPy import MindtPySolver
from pyomo.opt import SolverStatus, TerminationCondition

import random

def generate_random_initial_points():
    x_val = random.uniform(0.0, 2.0)  # Random value for x in [0.0, 2.0]
    y_val = random.choice([-1, 0, 1])  # Random integer value for y from {-1, 0, 1}
    return {'x': x_val, 'y': y_val}
random_initial_points = [generate_random_initial_points() for _ in range(10)]

model = ConcreteModel()

model.x = Var(domain=NonNegativeReals, bounds=(0.0, 2.0))
model.y = Var(domain=Integers, bounds=(-1, 1))



def f(x,y):
    return x**3 -3 * y * x**2 + 2 * x

# Stage 1 - Minimize loss
def loss_rule(m):
    return f(m.x, m.y)


model.loss = Objective(rule=loss_rule, sense=minimize)

model.loss = Objective(expr=f(model.x,model.y), sense=minimize)

solver = MindtPySolver()

for point in random_initial_points:
    print("=========================================")
    model.x.set_value(point['x'])
    model.y.set_value(point['y'])
    print(f"Initial x: {model.x()}")
    print(f"Initial y: {model.y()}")

    results = solver.solve(model, strategy='GOA',time_limit=13600,
                                    #    mip_solver= 'glpk', 
                                    # nlp_solver='ipopt',
                                    tee=True,
                                    use_mcpp = True,
                                    add_cuts_at_incumbent = True
                                    )
    
    print(f"Solver status: {results.solver.status}")
    print(f"Termination condition: {results.solver.termination_condition}")
    print(f"Optimal x: {model.x()}")
    print(f"Optimal y: {model.y()}")
    print(f"minimum loss: {model.loss()}")


# model.display()
# model.solutions.store_to(model)

# print(f"x=1.57,y=1:{f(1.57,1)}")
