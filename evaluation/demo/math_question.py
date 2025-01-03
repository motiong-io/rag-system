from pyomo.environ import *
from pyomo.contrib.mindtpy.MindtPy import MindtPySolver



model = ConcreteModel()

model.x = Var(domain=Reals, bounds=(-5, 5))
model.b = Var(domain=Binary)
model.y = Expression(expr=2 * model.b - 1)


# Stage 1 - Minimize loss
def loss_rule(m):

    return 3 * m.x**2 + 6 * m.y * m.x**2 + 2 * m.x

model.loss = Objective(rule=loss_rule, sense=minimize)


solver = MindtPySolver()
solver.solve(model, strategy='OA') 

print(f"Optimal x: {model.x()}")
print(f"Optimal y: {model.y()}")
print(f"minimum loss: {model.loss()}")
