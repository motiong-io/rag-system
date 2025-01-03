import re

from app.model.message_for_llm import MessageForLLM
from app.llm.get_llm import get_llm


def check_answer(question:str, ground_truth:str, generated_answer:str):
    llm = get_llm(llm_name='llama3_3')

    prompt = f"""
            Question: {question}
            Ground Truth: {ground_truth}
            Generated Answer: {generated_answer}
            Is the generated answer correct? Provide reasoning and respond with 'Y' or 'N'.

            Output format:
            before output
            <START>
            answer
            </END>
            other output
            """

    conversation = [
        MessageForLLM(
            role="user",
            content=prompt,
        )
        ]
    
    result = llm.full_response(conversation=conversation, sys_prompt="You are a helpful assistant to check if the answer is correct", temperature=0, frequency_penalty=0.0)
    match_result = re.search(r'<START>\s*["\']?([YN])["\']?\s*</END>', result).group(1)
    return match_result



def test_check_answer():
    question = "If my future wife has the same first name as the 15th first lady of the United States' mother and her surname is the same as the second assassinated president's mother's maiden name, what is my future wife's name? "
    ground_truth = "Jane Ballou"
    generated_answer = "To determine your future wife's name, we need to gather the required information from the observations.\n\nThe 15th first lady of the United States was Harriet Lane, and her mother's first name was Jane (Jane Ann Buchanan Lane). \n\nThe second assassinated president was James A. Garfield, and his mother's maiden name was Eliza Ballou, but we are interested in the surname which is Ballou.\n\nSo, if your future wife has the same first name as Harriet Lane's mother (Jane) and her surname is the same as James A. Garfield's mother's maiden name (Ballou), then your future wife's name would be Jane Ballou."

    check = check_answer(question, ground_truth, generated_answer)
    print(check)


if __name__ == "__main__":
    test_check_answer() 