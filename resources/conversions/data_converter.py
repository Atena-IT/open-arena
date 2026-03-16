import pandas as pd


""" CONFIG """
INPUT_FILE = "ToolScale.parquet"
OUTPUT_FILE = f"../data/ToolScale.xlsx"


""" MAIN """
if __name__ == "__main__":
    df = pd.read_parquet(INPUT_FILE)
    df.to_excel(OUTPUT_FILE, index=False)

    print(f"Conversion completed! File saved as {OUTPUT_FILE}")