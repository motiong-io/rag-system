import pandas as pd


def save_data(n:int):
    df = pd.read_csv("hf://datasets/google/frames-benchmark/test.tsv", sep="\t")

    df_first = df.head(n)

    df_first.to_csv(f"evaluation/dataset/row_{n}.csv", index=False)
    df_first.to_json(f"evaluation/dataset/row_{n}.json", orient="records",lines=True)


def read_data_row(row_index:int):
    df= pd.read_csv("evaluation/dataset/row_50.csv")
    # print(df.head())
    result = df.loc[df["Unnamed: 0"] == row_index, ["Prompt", "Answer","wiki_links"]].to_dict(orient="records")[0]
    # print(result)
    return result


def read_data_page(page_index:int, page_size:int=10):
    df= pd.read_csv("evaluation/dataset/row_50.csv")
    start_index = page_index * page_size
    end_index = start_index + page_size
    data = df.loc[start_index:end_index, ["Prompt", "Answer","wiki_links"]].to_dict(orient="records")
    return data


if __name__ == "__main__":
    save_data(50)
    # row=read_data_row(0)
    # print(row['Prompt'])
    # print(row['Answer'])
    