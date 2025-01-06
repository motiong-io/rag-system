import pandas as pd


def save_data():
    df = pd.read_csv("hf://datasets/google/frames-benchmark/test.tsv", sep="\t")

    df_first_30 = df.head(30)

    df_first_30.to_csv("evaluation/dataset/row_0_29.csv", index=False)
    df_first_30.to_json("evaluation/dataset/row_0_29.json", orient="records",lines=True)


def read_data_row(row_index:int):
    df= pd.read_csv("evaluation/dataset/row_0_29.csv")
    # print(df.head())
    result = df.loc[df["Unnamed: 0"] == row_index, ["Prompt", "Answer","wiki_links"]].to_dict(orient="records")[0]
    # print(result)
    return result


if __name__ == "__main__":
    # save_data()
    row=read_data_row(0)
    print(row['Prompt'])
    print(row['Answer'])
    