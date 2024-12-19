import json

from openai import OpenAI

from app.config import env

file_path = "benchmark/results.json"

llm = OpenAI(base_url=env.nvidia_base_url,
                            api_key=env.nvidia_api_key,
                            
                            )

def evaluate(correct_answer,generated_answer):
    response = llm.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "I will give you a correct answer and a generated answer, "
                    "use [yes/no] to help me tell if the generated answer is correct. "
                    "Do not output any other informations!"
                )
            },
            {
                "role": "user",
                "content": f"True Correct Answer: {correct_answer}"
            },
            {
                "role": "user",
                "content": f"Generated Answer: {generated_answer}"
            }
        ],
        model="nvidia/llama-3.1-nemotron-70b-instruct"
    )
    return response.choices[0].message.content

json_objects = []
current_json = ""
r_list=[]

with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        if line.strip():  
            current_json += line  
        else:  
            if current_json.strip(): 
                record=json.loads(current_json.strip())
                print(record['Answer'])
                # print(record['Result'])
                eval_result=evaluate(record['Answer'],record['Result'])
                r_list.append(eval_result)
                current_json = "" 
    if current_json.strip():
        record=json.loads(current_json.strip())
        print(record['Answer'])
        # print(record['Result'])
        eval_result=evaluate(record['Answer'],record['Result'])
        r_list.append(eval_result)


print(r_list)
count_yes=0
count_no=0
for i in r_list:
    if 'yes' in i.lower():
        count_yes+=1
    elif 'no' in i.lower():
        count_no+=1
    


print(f"Correctly generated answers: {count_yes}")
print(f"Incorrectly generated answers: {count_no}")
print(f"Total: {len(r_list)}")
print(f"Accuracy: {count_yes/len(r_list)}")

