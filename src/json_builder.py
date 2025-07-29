import json
import os
import re
from typing import List, Dict, Optional
from datetime import datetime

def save_qa_pairs(
    qa_pairs: List[Dict],
    output_dir: str,
    min_question_length: Optional[int] = None,
    min_answer_length: Optional[int] = None
) -> str:
    """
    Save Q&A pairs to JSON files with timestamp
    Format: [{"question": "...", "answer": "...", ...}]
    Optional: set min_question_length and min_answer_length (fallbacks to 10/15 if not given).
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"qa_pairs_{timestamp}.json")

    # Use provided or default values
    min_q_len = min_question_length if min_question_length is not None else 10
    min_a_len = min_answer_length if min_answer_length is not None else 15

    validated_pairs = []
    for pair in qa_pairs:
        # Basic validation
        if (pair.get("question") and pair.get("answer") and
            len(pair["question"]) > min_q_len and 
            len(pair["answer"]) >= min_a_len):

            # Clean up formatting
            question = re.sub(r'\s+', ' ', pair["question"]).strip()
            answer = re.sub(r'\s+', ' ', pair["answer"]).strip()

            # Make sure question ends with question mark
            if not question.endswith('?'):
                question = question + '?'

            validated_pairs.append({
                "question": question,
                "answer": answer,
                "source": pair.get("source", "unknown"),
                "confidence": pair.get("confidence", 0.0)
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(validated_pairs, f, indent=2, ensure_ascii=False)

    return output_path