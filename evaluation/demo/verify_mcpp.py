from pyomo.contrib.mcpp.pyomo_mcpp import _MCPP_lib

try:
    _MCPP_lib().get_version()
    print("MCPP is installed and accessible.")
except AttributeError as e:
    print("MCPP installation failed:", e)
