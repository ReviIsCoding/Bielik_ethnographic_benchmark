import google.generativeai as genai
from dotenv import load_dotenv
import os
import pandas as pd
import time
import json

def test_llm(test_data_path: str, llm_version: str, llm_id: str, key: str):
    # konfiguracja google api

    if key == "GOOGLE_API_KEY":
        load_dotenv()
        GOOGLE_API_KEY = os.getenv(key)
        genai.configure(api_key=GOOGLE_API_KEY)

    else:
        print("Błąd konfiguracji api.")


    # wczytanie modelu, sprawdzić jak dla innych
    version = llm_version
    model = genai.GenerativeModel(version)


    # wczytanie danych
    #df = pd.read_csv(test_data_path)
    df = pd.read_excel(test_data_path, sheet_name='test')
    print(df)

    # testowanie modelu
    responses = []
    for index, row in df.iterrows():

        prompt = f"""Korzystając z wiedzy, którą masz, odpowiedz na pytanie: {row['Pytanie']}.
Do wyboru masz 4 odpowiedzi:
A: {row['A']},
B: {row['B']},
C: {row['C']},
D: {row['D']}.   

W odpowiedzi zwróć tylko literę A, B, C lub D odpowiedzi, którą uważasz za poprawną.
"""

        # czekanie, bo limity gemini
        if (index + 1) % 15 == 0:
            print("Waiting")
            time.sleep(60)


        response = model.generate_content(prompt)
        answer = response.text
        responses.append(answer)

    # zapis do .csv
    df['answers'] = responses
    #df.to_csv(output_path, index=False)
    #df.to_excel(output_path, sheet_name='Sheet1')

    answers_list = df.to_dict(orient="records")
    # json - llm_id, 
    # Podsumowanie - json z liczbą pytań, poprawnych, błędnych
    output_json = {
        "id modelu": llm_id,
        "podsumowanie": {
            "liczba pytań": df.shape[0],
            "prowidłowe odpowiedzi": 7
        },
        "liczba_pytań": df.shape[0],
        "odpowiedzi": answers_list
    }
    # odpowiedzi - lista pytań


    folder = "results"
    filename = f"{llm_id}.jsonl"
    filepath = os.path.join(folder, filename)
    os.makedirs(folder, exist_ok=True)
    df.to_json(filepath, orient="records", lines=True, force_ascii=False)

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)



input_path = "input.xlsx"
llm_id = "gemini-1.5-flash"
llm_name = "gemini-1.5-flash"
llm_version = 'models/gemini-1.5-flash'
key = "GOOGLE_API_KEY"


test_llm(input_path, llm_version, llm_id, key)