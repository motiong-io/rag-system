from evaluation.dataset import read_data_row

from evaluation.services.evaluation_steps import EvaluationSteps


# benchmark data read
def get_qa(row_index:int):
    row=read_data_row(row_index)
    return row['Prompt'], row['Answer']

# evaluation
def calculate_loss(evaluation_steps:EvaluationSteps):
    """
    Calculate the loss of the model on the given row of the benchmark.
    """
    record = []
    for row_index in range(30): 
        print(f"========= Row index {row_index} =======")
        question, ground_truth = get_qa(row_index)
        check_result = evaluation_steps.run(question, ground_truth)
        record.append(check_result)
    
    total_count = len(record)
    n_count = record.count('N')
    return_loss = n_count/total_count
    print(f"Total count: {total_count}")
    print(f"Number of incorrect answers: {n_count}")
    print(f"Loss: {return_loss}")
    return return_loss

