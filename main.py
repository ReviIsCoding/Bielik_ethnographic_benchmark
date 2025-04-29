import google.generativeai as genai
from dotenv import load_dotenv
import os
import pandas as pd
import time
import json

def main(test_data_path: str, llm_version: str, llm_id: str, key: str):
    # konfiguracja google api

    if key == "GOOGLE_API_KEY":
        load_dotenv()
        GOOGLE_API_KEY = os.getenv(key)
        genai.configure(api_key=GOOGLE_API_KEY)

        # wczytanie modelu, sprawdzić jak dla innych
        version = llm_version
        model = genai.GenerativeModel(version)

    if key == "OPENAI_API_KEY":
        print("Konfiguracja Open AI")

    else:
        print("Błąd konfiguracji api.")


    # wczytanie danych
    #df = pd.read_csv(test_data_path)
    df = pd.read_excel(test_data_path, sheet_name='test')
    print(df)

    # testowanie modelu
    responses = []
    correctness = []
    for index, row in df.iterrows():

        print(f"Odpowieadam na pytanie {index+1}")
        prompt = f"""Korzystając z wiedzy, którą masz, odpowiedz na pytanie: {row['Pytanie']}.
Do wyboru masz 4 odpowiedzi:
A: {row['A']},
B: {row['B']},
C: {row['C']},
D: {row['D']}.   
W odpowiedzi podaj tylko literę A, B, C lub D odpowiedzi, którą uważasz za poprawną.
"""

        # czekanie, bo limity gemini 
        if key == "GOOGLE_API_KEY" and (index + 1) % 15 == 0:
            print("Waiting")
            time.sleep(60)

        #response = model.generate_content(prompt)
        response = model.generate_content(
            contents=prompt,
            generation_config={
                "max_output_tokens": 1
            }
        )
        answer = response.text
        responses.append(answer)

        if answer == row['Pozycja']:
            correctness.append(1)
        else:
            correctness.append(0)

    # zapis do .csv
    df['answers'] = responses
    df['correctness'] = correctness
    #df.to_csv(output_path, index=False)
    #df.to_excel(output_path, sheet_name='Sheet1')

    # obliczenie liczby poprawnych odpowiedzi
    correct_answers = df['correctness'].sum()
    correct_answers = int(correct_answers)



    # Podsumowanie - json z liczbą pytań, poprawnych, błędnych
    answers_list = df.to_dict(orient="records")
    output_json = {
        "id modelu": llm_id,
        "liczba_pytań": df.shape[0],
        "podsumowanie": {
            "prowidłowe odpowiedzi": correct_answers
        },
        "odpowiedzi": answers_list
    }

    # dodatkowo lista jsonl z pytaniami
    folder = "results"
    filename = f"{llm_id}.jsonl"
    filepath = os.path.join(folder, filename)
    os.makedirs(folder, exist_ok=True)
    df.to_json(filepath, orient="records", lines=True, force_ascii=False)

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)

    print("Zapisano")


if __name__=="__main__":
    main(test_data_path = "input.xlsx", 
         llm_version = "models/gemini-1.5-flash", 
         llm_id = "gemini-1.5-flash", 
         key = "GOOGLE_API_KEY")
