# 📘 Bielik Ethnographic Benchmark

Testowanie polskich modeli językowych w kontekście wiedzy etnologicznej i kultury lokalnej.

---

## 🔍 Cel projektu

Stworzenie benchmarku porównującego jakość odpowiedzi różnych modeli językowych (np. GPT-4, Gemini, Bielik) na pytania zamknięte z zakresu etnologii i historii społecznej w Polsce.

---

## 🗂️ Struktura repozytorium

```
├── benchmark_test_llm_main.py    # Główny skrypt uruchamiający testowanie
├── results/                      # Folder z odpowiedziami modeli i statystykami
├── tests/                        # Folder z plikami testów jednostkowych i integracyjnych
├── moduły/                       # Folder z modułami funkcjonalnymi
│   ├── dataset_loader.py         # Wczytywanie danych testowych z pliku CSV/XLSX
│   ├── llm_connector.py          # Obsługa komunikacji z modelami (lokalnie/API)
│   ├── response_saver.py         # Zapis wyników do JSON/CSV
│   └── utils.py                  # Funkcje pomocnicze (parsowanie outputu, budowa promptu)
├── _natalia_prototyp/           # Archiwum pierwszej wersji benchmarku
├── requirements.txt             # Lista wymaganych bibliotek
└── README.md                    # Niniejszy plik
```

---

## ▶️ Uruchamianie benchmarku - generowanie odpowiedzi

### Wersja CLI:

```bash
python benchmark_test_llm_main.py \
  --llm="bielik-chat" \
  --llm-name="Bielik" \
  --test="./test_files/test.xlsx" \
  --results="./results/bielik.json" \
  --api="local" \
  --interval=0
```
Po uruchomieniu benchmarku zapisywana jest lista surowych odpowiedzi modelu. 
Porównanie odpowiedzi z prawidłowymi i statystyki są generowane w osobnym kroku (skrypt `benchmark_merge_results.py`).

### Wymagane argumenty:

- `--llm` – unikalny identyfikator modelu
- `--llm-name` – przyjazna nazwa modelu
- `--test` – ścieżka do pliku testowego (`.csv` lub `.xlsx`)

### Opcjonalne:

- `--results` – ścieżka do pliku wyjściowego
- `--api` – typ API (`local`, `openAI`, `vllm`)
- `--url`, `--key` – jeśli używasz modelu przez API (np. OpenAI)
- `--interval` – opóźnienie między zapytaniami

---

## 🧪 Dane wejściowe (testy)

Plik testowy powinien zawierać kolumny:

- `Pytanie`
- `A`, `B`, `C`, `D` – możliwe odpowiedzi
- `Pozycja` – poprawna odpowiedź (litera A-D)
- `Domena`, `Kategoria`, `Tagi`

---

## 📤 Dane wyjściowe

Po uruchomieniu benchmarku zapisuje:

- `results/model_raw.json` – surowe odpowiedzi modelu na każde pytanie (bez oceny)
- (w kolejnym kroku) `results/model_summary.json` – podsumowanie ocen (tworzone osobnym skryptem)

---

## 📝 Tworzenie promptu i przetwarzanie odpowiedzi

- Prompt budowany jest na podstawie każdego wiersza z pliku testowego, zgodnie z szablonem zdefiniowanym w `utils.py` (`PROMPT_TEMPLATE`).
- Odpowiedzi modelu są parsowane funkcją `parse_output()` z `utils.py` i zapisywane w surowej formie do pliku JSON przez `response_saver.py`.
- Ocena poprawności i podsumowanie wyników odbywa się w kolejnym kroku, przez osobny skrypt (`benchmark_merge_results.py`).

## 🧪 Testowanie

Projekt zawiera dwa poziomy testów:

- **Testy jednostkowe** – dotyczą funkcji pomocniczych z modułów w folderze `modules/` (np. `utils`, `dataset_loader`, `response_saver`). Znajdują się w `tests/unit/`.
- **Testy integracyjne** – obejmują główny przebieg działania benchmarku:
  - `benchmark_test_llm_main.py` – testowanie logiki uruchamiania modelu
  - `benchmark_merge_results.py` – testowanie przetwarzania wyników i generowania statystyk  
  Znajdują się w `tests/integration/`.

### 🔄 Uruchamianie testów

Uruchomienie wszystkich testów w bash:
`pytest`

Tylko testy jednostkowe w bash:
`pytest tests/unit/`

Tylko testy integracyjne w bash:
`pytest tests/integration/`

## 🔧 Wymagania

- Python 3.9+
- Biblioteki: `pandas`, `openai`, `google-generativeai`, `argparse`, `dotenv`

Instalacja:

```bash
pip install -r requirements.txt
```

---

## 👥 Zespół

- Natalia Nadolna (https://github.com/NataliaNadolna)
- Anna Zielińska (https://github.com/ReviIsCoding)
- Krzysztof Raszczuk – konsultacje merytoryczne

---
