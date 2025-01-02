from pyomo.environ import *
from pyomo.contrib.mindtpy.MindtPy import MindtPySolver



model = ConcreteModel()

model.x = Var(domain=Reals, bounds=(0, 2))
model.y = Var(domain=Integers, bounds=(-1, 1))



def f(x,y):
    return x**3 -3 * y * x**2 + 2 * x

# Stage 1 - Minimize loss
def loss_rule(m):
    return f(m.x, m.y)

# def calc():
#     x = 1.57
#     y = 1
#     return f(x,y)

# print(calc())

model.loss = Objective(rule=loss_rule, sense=minimize)


solver = MindtPySolver()
solver.solve(model, strategy='GOA',time_limit=3600,
                                #    nlp_solver='ipopt',
                                   tee=True)

print(f"Optimal x: {model.x()}")
print(f"Optimal y: {model.y()}")
print(f"minimum loss: {model.loss()}")

model.display()