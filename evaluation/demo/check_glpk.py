from pyomo.environ import *
glpk_solver = SolverFactory("glpk")
glpk_solver.options["mipgap"] = 0.01
glpk_solver.options["tmlim"] = 60
glpk_solver.options["presolve"] = 2
glpk_solver.options["threads"] = 4

print(glpk_solver.options)