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
├── test_files/                   # Przykładowe pliki testowe (CSV/XLSX)
├── moduły/                       # Folder z modułami funkcjonalnymi
│   ├── dataset_loader.py         # Wczytywanie danych testowych
│   ├── llm_connector.py          # Obsługa komunikacji z modelami (lokalnie/API)
│   ├── response_saver.py         # Zapis wyników do JSON/CSV
│   └── utils.py                  # Funkcje pomocnicze
├── _natalia_prototyp/           # Archiwum pierwszej wersji benchmarku
├── requirements.txt             # Lista wymaganych bibliotek
└── README.md                    # Niniejszy plik
```

---

## ▶️ Uruchamianie benchmarku

### Wersja CLI:

```bash
python main.py \
  --llm="bielik-chat" \
  --llm-name="Bielik" \
  --test="./test_files/test.xlsx" \
  --results="./results/bielik.json" \
  --api="local" \
  --interval=0
```

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

Skrypt zapisuje dwa pliki:

1. `results/model.jsonl` – lista odpowiedzi modelu na każde pytanie
2. `output.json` – podsumowanie (liczba pytań, poprawnych odpowiedzi itd.)

---

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
- Revi (https://github.com/ReviIsCoding)
- Krzysztof Raszczuk – konsultacje merytoryczne

---
