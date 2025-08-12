import re
from typing import Tuple

ANSWER_RE = re.compile(r'answer\s*:\s*\[?\s*([ABCD])\s*\]?', re.IGNORECASE)
EXPL_RE   = re.compile(r'explanation\s*:\s*(.+)', re.IGNORECASE | re.DOTALL)

def parse_output(raw_output: str) -> Tuple[str, str]:
    """
    Parse output. Succeed only if we have BOTH:
    - an answer letter A–D (explicit or fallback),
    - an Explanation: <text>.
    Otherwise return the standard parsing error tuple.
    """
    try:
        # 1) answer: explicit "Answer: X" OR fallback standalone A–D
        m_answer = ANSWER_RE.search(raw_output)
        if not m_answer:
            m_answer = re.search(r'\b([ABCD])\b', raw_output, flags=re.IGNORECASE)

        # 2) explanation: must be an "Explanation: ..." section
        m_expl = EXPL_RE.search(raw_output)

        # 3) require BOTH; otherwise -> parsing error
        if not m_answer or not m_expl:
            return "Parsing error", "Exception during parsing."

        answer = m_answer.group(1).upper()
        explanation = m_expl.group(1).strip()

        return answer, explanation

    except Exception as e:
        print(f"Error parsing output: {e}")
        return ("Parsing error", "Exception during parsing.")
    
PROMPT_TEMPLATE = (
    """Wybierz poprawną odpowiedź spośród A, B, C i D. Uzasadnij krótko swój wybór.

    Podaj wynik WYŁĄCZNIE w tym formacie:
    Answer: [A/B/C/D]
    Explanation: [krótka przyczyna]

    Przykład:
    Answer: [C]
    Explanation: Krótko dlaczego C.

    Pytanie: {question}
    A: {A}
    B: {B}
    C: {C}
    D: {D}
    """
)
    
def build_prompt(row) -> str:
    """Builds a prompt for the model from a DataFrame row.
        
    Args:
        row (pd.Series): A row from the DataFrame with columns 'Pytanie', 'A', 'B', 'C', 'D'.
            
    Returns:
       str: Formatted prompt string.
    """
    return PROMPT_TEMPLATE.format(
         question=row['Pytanie'],
         A=row['A'],
         B=row['B'],
         C=row['C'],
         D=row['D']                                                                       
    )