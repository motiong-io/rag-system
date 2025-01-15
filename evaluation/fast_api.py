import sys
import uvicorn
from fastapi import FastAPI
import io

from evaluation.calculate_loss import evaluate_in_threads
from evaluation.request_model import EvaluateRagRequest

from evaluation.results.record import ResultRecordService


app = FastAPI()

# 捕获 stdout
def capture_stdout(func, *args, **kwargs):
    old_stdout = sys.stdout  # 保存原 stdout
    new_stdout = io.StringIO()  # 创建一个新的缓冲区
    sys.stdout = new_stdout  # 替换 stdout 为缓冲区
    try:
        result = func(*args, **kwargs)  # 调用目标函数
    finally:
        sys.stdout = old_stdout  # 恢复原 stdout
    output = new_stdout.getvalue()  # 获取缓冲区内容
    return result, output


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/evaluate_rag/")
def evaluate_rag(request: EvaluateRagRequest):
    rrs = ResultRecordService()
    result = evaluate_in_threads(request.indices, request.N_s, request.N_r, request.alpha, 
            request.T, request.P_f, request.MSR, request.CE, request.max_workers

    )
    rrs.add_contents(request.model_dump())
    rrs.add_contents(result)
    
    return result

if __name__ == "__main__":
    #http://172.28.195.191:8503
    uvicorn.run(app, host="0.0.0.0", port=8503)
    # result_record = ResultRecordService("results.txt")
    # result_record.add_contents("Hello")

