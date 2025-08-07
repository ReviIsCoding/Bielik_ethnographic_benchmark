# 📘 Bielik Ethnographic Benchmark

Testowanie polskich modeli językowych w kontekście wiedzy etnologicznej i kultury lokalnej.

---

## 🔍 Cel projektu

Stworzenie benchmarku porównującego jakość odpowiedzi różnych modeli językowych (np. GPT-4, Gemini, Bielik) na pytania zamknięte z zakresu etnologii i historii społecznej w Polsce.

---

## 🧠 Architektura i logika działania

Projekt jest modularny i rozdzielony na logiczne komponenty:

### 🔹 Struktura backendów (komunikacja z modelami)
- `llm_connector.py` – główny punkt wejścia: funkcja `ask_model(config)` deleguje zapytanie do odpowiedniego backendu.
- `local_backend.py` – obsługa modeli lokalnych (np. Bielik z Hugging Face Transformers).
- `api_backend.py` – obsługa modeli przez API (OpenAI, Gemini, Hugging Face Inference API).

Backend wybierany jest dynamicznie na podstawie pola `api` w `model_config`.

### 🔹 Konfiguracja modelu
- Wszystkie parametry modelu (id, typ API, długość odpowiedzi, URL, klucz API, kwantyzacja) przekazywane są przez argumenty CLI i trafiają do jednej struktury: `model_config`.

### 🔹 Obsługa wyjątków
- Każdy backend posiada własną obsługę błędów (brak odpowiedzi, timeouty, złe dane wejściowe, błędne API key itp.).
- Główna pętla benchmarku nie przerywa działania w przypadku błędu jednego zapytania.

### 🔹 Raportowanie
- Czas wykonania benchmarku i liczba przetworzonych pytań są wypisywane po zakończeniu działania.
- Wyniki zapisywane są do pliku `.jsonl` (lista odpowiedzi) i `.json` (podsumowanie).

---


## 🗂️ Struktura repozytorium

```
├── benchmark_test_llm_main.py        # Główny skrypt uruchamiający testowanie modeli
├── benchmark_merge_results.py        # Skrypt scalający i oceniający odpowiedzi modeli
│
├── moduły/                           # Główne komponenty systemu
│   ├── dataset_loader.py             # Wczytywanie danych testowych z pliku CSV/XLSX
│   ├── llm_connector.py              # Delegator: wybiera odpowiedni backend w zależności od konfiguracji
│   ├── local_backend.py              # Obsługa modeli lokalnych (np. Hugging Face, Bielik)
│   ├── api_backend.py                # Obsługa modeli przez API (OpenAI, Gemini, HF Inference API)
│   ├── response_saver.py             # Zapis wyników do JSON/JSONL
│   └── utils.py                      # Funkcje pomocnicze (parsowanie outputu, budowa promptu)
│
├── results/                          # Folder z odpowiedziami modeli i podsumowaniami
│   ├── <llm_id>_raw.json             # Surowe odpowiedzi modelu (JSON lub JSONL)
│   └── <llm_id>_summary.json         # Podsumowanie poprawności odpowiedzi
│
├── tests/                            # Testy jednostkowe i integracyjne
│   ├── unit/                         # Testy funkcji pomocniczych i backendów
│   └── integration/                  # Testy pełnych przepływów działania skryptów
│
├── _natalia_prototyp/               # Archiwum pierwszej wersji benchmarku (nieużywane)
│
├── requirements.txt                 # Lista wymaganych bibliotek
└── README.md                        # Niniejszy plik
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
  --api="local"  # dostępne: local, openAI, google, hf_api
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

---
## 🧪 Testowanie

Projekt zawiera pokrycie testami jednostkowymi i integracyjnymi:

### ✅ Testy jednostkowe
- Dla każdego backendu (`local_backend`, `api_backend`, `llm_connector`).
- Obejmują:
  - poprawność działania delegacji,
  - obsługę wyjątków (np. błędne configi, brak API key, błędy sieci),
  - parsowanie odpowiedzi,
  - przekazywanie argumentów.

### ✅ Testy integracyjne
- `benchmark_test_llm_main.py` – testuje pełen przebieg generowania odpowiedzi.


### 🧪 Przypadki brzegowe
Testy obejmują m.in.:
- brak wymaganych pól w `model_config`,
- nieobsługiwany typ API (`invalid_api`),
- wyjątki rzucane przez modele (np. `TimeoutError`, `OpenAIError`),
- nieoczekiwany format odpowiedzi (np. brak `choices[0].message.content`).

### 🔧 Mockowanie
- Testy używają `mock` i `monkeypatch`, co pozwala symulować zachowanie modeli bez realnego API.

### 📝 Czytelność
- Każdy test posiada `docstring` z opisem celu testu.
- Pliki testowe znajdują się w folderze `tests/`.

---

## ✅ Status projektu

- ✅ Modularna architektura backendów
- ✅ Obsługa: `local`, `openAI`, `google`, `hf_api`
- ✅ Czytelna struktura promptów i wyników
- ✅ Obsługa wyjątków i błędów sieciowych
- ✅ Pokrycie testami jednostkowymi i integracyjnymi
- 🔜 Planowane: scalony raport porównawczy dla wielu modeli (`benchmark_merge_results.py`)

---

## 🔧 Wymagania

- Python 3.9+
- Biblioteki: `pandas`, `openai`, `google-generativeai`, `argparse`, `dotenv`, `requests`

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
