from typing import List,Literal
from pydantic import BaseModel

class EvaluateRagRequest(BaseModel):
    indices: List[int]
    N_s: int
    N_r: int
    alpha: float
    T: float
    P_f: float
    MSR: Literal[0,1]
    CE: Literal[0,1]
    max_workers: int = 4