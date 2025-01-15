from evaluation.dataset import read_data_row
from concurrent.futures import ProcessPoolExecutor, as_completed
from evaluation.services.evaluation_steps import EvaluationSteps
from typing import Literal
from pqdm.processes import pqdm
from evaluation.results.record import ResultRecordService

rrs = ResultRecordService()
# benchmark data read
def get_qa(row_index:int):
    row=read_data_row(row_index)
    return row['Prompt'], row['Answer']

# evaluation
def calculate_loss(
        N_s: int,
        N_r: int,
        alpha: float,
        T: float,
        P_f: float,
        MSR: Literal[0, 1],
        CE: Literal[0, 1]
):
    """
    Calculate the loss of the model on the given row of the benchmark.
    """
    evaluation_steps = EvaluationSteps(
    number_chunks_limit=N_s,
    hybrid_search_alpha=alpha,
    number_chunks_rerank= N_r,
    temperature_LLM= T,
    penalty_frequency_LLM = P_f,
    if_multi_step_RAG= MSR,
    if_contextual_embedding = CE
)

    record = []
    for row_index in range(30): 
        print(f"========= Row index {row_index} =======")
        question, ground_truth = get_qa(row_index)
        check_result = evaluation_steps.run(question, ground_truth)
        record.append(check_result)
    

    evaluation_steps.context_provider.close()
    
    total_count = len(record)
    n_count = record.count('N')
    return_loss = n_count/total_count
    print(f"Total count: {total_count}")
    print(f"Number of incorrect answers: {n_count}")
    print(f"Loss: {return_loss}")
    return return_loss


def evaluate_thread(index, N_s, N_r, alpha, T, P_f, MSR, CE):
    question, ground_truth = get_qa(index)
    eva = EvaluationSteps(N_s, alpha, N_r, T, P_f, MSR, CE)
    check_result = eva.run(question, ground_truth)
    eva.context_provider.close()
    return check_result



## for fast_api
def evaluate_in_threads(indices:list, N_s, N_r, alpha, T, P_f, MSR, CE, max_workers=4):
    """
    多线程调用 evaluate_thread 函数。

    :param indices: 需要评估的索引列表。
    :param N_s: 参数。
    :param N_r: 参数。
    :param alpha: 参数。
    :param T: 参数。
    :param P_f: 参数。
    :param MSR: 参数。
    :param CE: 参数。
    :param max_workers: 最大线程数。
    :return: 每个索引的结果字典。
    """
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(evaluate_thread, index, N_s, N_r, alpha, T, P_f, MSR, CE): index
            for index in indices
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
                # print("==============================")
                rrs.add_contents(f"Index {index} finished.")
                print(f" {len(results)} / {len(indices)} finished.")
            except Exception as e:
                print(f"Index {index} generated an exception: {e}")
                rrs.add_contents(f"Index {index} generated an exception: {e}")
                results[index] = None
    return results


def calculate_loss_threaded(
        N_s: int,
        N_r: int,
        alpha: float,
        T: float,
        P_f: float,
        MSR: Literal[0, 1],
        CE: Literal[0, 1]
):
    """
    Calculate the loss of the model on the given row of the benchmark.
    """
    indices = list(range(8))
    results = evaluate_in_threads(indices, N_s, N_r, alpha, T, P_f, MSR, CE)
    total_count = len(results)
    n_count = list(results.values()).count('N')
    return_loss = n_count / total_count
    print(f"Total count: {total_count}")
    print(f"Number of incorrect answers: {n_count}")
    print(f"Loss: {return_loss}")
    return return_loss



async def a_calculate_loss(
        N_s: int,
        N_r: int,
        alpha: float,
        T: float,
        P_f: float,
        MSR: Literal[0, 1],
        CE: Literal[0, 1]
):
    """
    Calculate the loss of the model on the given row of the benchmark.
    """
    evaluation_steps = EvaluationSteps(
    number_chunks_limit=N_s,
    hybrid_search_alpha=alpha,
    number_chunks_rerank= N_r,
    temperature_LLM= T,
    penalty_frequency_LLM = P_f,
    if_multi_step_RAG= MSR,
    if_contextual_embedding = CE
)

    record = []
    for row_index in range(30): 
        print(f"========= Row index {row_index} =======")
        question, ground_truth = get_qa(row_index)
        check_result = await evaluation_steps.a_run(question, ground_truth)
        record.append(check_result)
    
    evaluation_steps.context_provider.close()
    
    total_count = len(record)
    n_count = record.count('N')
    return_loss = n_count/total_count
    print(f"Total count: {total_count}")
    print(f"Number of incorrect answers: {n_count}")
    print(f"Loss: {return_loss}")
    return return_loss








if __name__ == '__main__':
    # calculate_loss(30, 10, 0.5, 0.1, 0.5, 1, 0)
    calculate_loss_threaded(30, 10, 0.5, 0.1, 0.5, 1, 0)

    # asyncio.run(a_calculate_loss(30, 10, 0.5, 0.1, 0.5, 1, 0))