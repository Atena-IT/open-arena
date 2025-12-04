import json, time
import pandas as pd


""" CONFIG """
INPUT_FILE = "input_notions.xlsx"
OUTPUT_FILE = f"../data/output_{time.time()}.xlsx"


""" MAIN """
if __name__ == "__main__":

    # Reading Excel input file
    df = pd.read_excel(INPUT_FILE)
    print(f"Read '{INPUT_FILE}' file successfully")

    # Creating 'Metadata' column in JSON format
    df['Metadata'] = df.apply(lambda row: json.dumps(
        {"metadata": [{'level': str(row['livello'])}, {'practical': str(row['pratico'])}]},
        ensure_ascii=False
    ), axis=1)

    # Creating new dataframe
    df_new = pd.DataFrame({
        'ID': df['ID'],
        'Topic': df['Argomento'],
        'Question': df['Domanda'],
        'Option A': df['opzione A'],
        'Option B': df['opzione B'],
        'Option C': df['opzione C'],
        'Option D': df['opzione D'],
        'Answer': df['Risposta'],
        'Metadata': df['Metadata']
    })

    # Storing Excel output file
    df_new.to_excel(OUTPUT_FILE, index=False)
    print(f"Stored '{OUTPUT_FILE}' file successfully!")
