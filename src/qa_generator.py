import re
import logging
from typing import List, Dict

import torch
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UniversalQAGenerator:
    def __init__(self, config: Dict):
        self.qa_settings = config.get("qa_settings", {})
        self.min_question_length = self.qa_settings.get("min_question_length", 10)
        self.min_answer_length = self.qa_settings.get("min_answer_length", 15)
        self.mode = self.qa_settings.get("mode", "auto")
        self.model = self._load_model() if self.mode in ["auto", "generate"] else None
        self.max_length = 512  # Default max sequence length
        self.stride = 128  # Reduced stride to prevent errors

    def _load_model(self):
        try:
            return pipeline(
                "question-answering",
                model="deepset/roberta-large-squad2",
                device=0 if torch.cuda.is_available() else -1,
                tokenizer_kwargs={
                    'max_length': self.max_length,
                    'stride': self.stride,
                    'truncation': True
                }
            )
        except Exception as e:
            logger.error(f"QA model loading failed: {str(e)}")
            return None

    def _extract_explicit_qa(self, text: str) -> List[Dict]:
        """For documents with explicit Q&A pairs"""
        qa_pairs = []
        def is_valid_answer(answer):
            return (
                    len(answer) >= self.min_answer_length and
                    '?' not in answer and  # Answers shouldn't contain questions
                    not re.search(r'(Q:|Question\s*\d)', answer)  # No next question markers
            )
        # More patterns to match various Q&A formats
        qa_patterns = [
            r'(?:Q:|Question\s*\d*[.:])\s*(.+?\?)\s*(?:A:|Answer\s*\d*[.:])\s*((?:(?!Q:|Question).)+?)(?=\s*(?:Q:|Question|\Z))',
            # Numbered questions with answer validation
            r'(?:\d+[.)]\s*)([^.?]+\?)\s*([^.?]+?(?:\n(?!\d+[.)]).+?)*)(?=\n\s*\d+[.)]|\Z)'
        ]

        for pattern in qa_patterns:
            for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
                question = match.group(1).strip()
                answer = match.group(2).strip()

                # More lenient validation
                if question and answer and ('?' in question or 'what' in question.lower() or 'how' in question.lower()):
                    min_length = self.qa_settings.get("min_answer_length", 10)
                    if len(answer) >= min_length:
                        qa_pairs.append({
                            "question": question,
                            "answer": answer,
                            "source": "extracted",
                            "confidence": 1.0
                        })
        if qa_pairs:
            return qa_pairs
        questions_list = []
        list_patterns = [
            r'(?:\d+\.\s*)([A-Z][^.?]+\?)',  # Numbered questions like "1. What is...?"
            r'(?:•|\*|-)\s*([A-Z][^.?]+\?)',  # Bulleted questions
            r'(?:\n|^)([A-Z][^.?]+\?)'  # Questions starting on new lines
        ]
        for pattern in list_patterns:
            questions_list.extend(re.findall(pattern, text))
        if self.mode == "extract" or not self.model:
            return []
        qa_pairs = []
        for question in questions_list[:10]:  # Limit to first 10 questions
            try:
                max_context_length = 384  # A safe value for most transformer models
                if len(text) > max_context_length * 3:  # If text is very long
                    # Create overlapping chunks
                    chunks = []
                    chunk_size = max_context_length * 2
                    overlap = max_context_length // 2

                    for i in range(0, len(text), chunk_size - overlap):
                        chunk = text[i:i + chunk_size]
                        if len(chunk) > 100:  # Only add substantial chunks
                            chunks.append(chunk)
                    best_result = None
                    best_score = 0
                    for chunk in chunks:
                        try:
                            result = self.model(question=question, context=chunk)
                            if result["score"] > best_score:
                                best_score = result["score"]
                                best_result = result
                        except Exception as chunk_err:
                            logger.debug(f"Error processing chunk: {str(chunk_err)}")
                            continue
                    result = best_result
                else:
                    # Process the full text if it's not too long
                    result = self.model(question=question, context=text)
                if result and result["score"] >= self.qa_settings.get("confidence_threshold", 0.65):
                    if (question and result["answer"] and
                        len(question) > self.min_question_length and
                        len(result["answer"]) >= self.min_answer_length):
                        qa_pairs.append({
                            "question": question,
                            "answer": result["answer"],
                            "source": "model_extracted",
                            "confidence": round(result["score"], 2)
                        })
            except Exception as e:
                logger.error(f"Error generating answer for question: {question}. Error: {str(e)}")
        valid_pairs = []
        for pair in qa_pairs:
            if is_valid_answer(pair['answer']):
                pair['confidence'] = 1.0  # Keep original confidence
                valid_pairs.append(pair)
        return valid_pairs

    def _generate_qa_from_content(self, text: str) -> List[Dict]:
        """For documents without explicit Q&A"""
        if not self.model or not text:
            return []
        confidence_threshold = max(0.3, self.qa_settings.get("confidence_threshold", 0.6) - 0.15)
        try:
            chunks = []
            for i in range(0, len(text), 1500):
                chunk = text[i:i + 2000]
                if len(chunk) > 200:  # Only process substantial chunks
                    chunks.append(chunk)
            if not chunks:
                chunks = [text[:2000]]  # Fallback to first 2000 chars
            logger.info(f"Processing {len(chunks)} text chunks for QA generation")
            # Define common question templates that work well with QA models
            generic_questions = [
                "What is the main topic discussed in this text?",
                "What are the key points mentioned?",
                "How does this text describe the main concept?",
                "What is the definition provided in this text?",
                "What are the benefits mentioned in this text?",
                "What challenges are discussed in this text?",
                "What is the most important information in this text?",
                "What are the steps or process described?",
                "How does this text explain the concept?",
                "What are the requirements mentioned in this text?"
            ]

            qa_pairs = []
            # Lower confidence threshold for better results
            confidence_threshold = self.qa_settings.get("confidence_threshold", 0.6)
            # Reduce threshold by 0.2 to get more results
            adjusted_threshold = max(0.3, confidence_threshold - 0.2)

            # Try each generic question against each chunk
            for question in generic_questions[:5]:  # Limit to first 5 questions for efficiency
                best_result = None
                best_score = 0

                for chunk in chunks[:5]:  # Limit to first 5 chunks for efficiency
                    try:
                        result = self.model(question=question, context=chunk)
                        if result and result["score"] > best_score:
                            best_score = result["score"]
                            best_result = result
                    except Exception as e:
                        logger.debug(f"Error processing chunk: {str(e)}")
                        continue

                if best_result and best_result["score"] >= adjusted_threshold:
                    # Clean up the answer - remove very short answers
                    answer = best_result["answer"].strip()
                    if len(answer) >= self.qa_settings.get("min_answer_length", 15):
                        qa_pairs.append({
                            "question": question,
                            "answer": answer,
                            "source": "generated",
                            "confidence": round(best_result["score"], 2)
                        })

            # Also try to extract keyphrases from the text for more specific questions
            try:
                from nltk.corpus import stopwords
                from nltk.tokenize import word_tokenize

                # Extract potential key terms
                stop_words = set(stopwords.words('english'))
                words = word_tokenize(text[:10000])  # Process first 10K chars
                filtered_words = [w for w in words if w.isalnum() and len(w) > 4 and w.lower() not in stop_words]

                # Count occurrences
                from collections import Counter
                word_counts = Counter(filtered_words)

                # Get top 3 terms
                top_terms = [term for term, count in word_counts.most_common(3) if count > 1]

                # Generate questions for these terms
                for term in top_terms:
                    question = f"What does this text say about {term}?"
                    best_result = None
                    best_score = 0

                    for chunk in chunks[:3]:  # Check first 3 chunks
                        try:
                            result = self.model(question=question, context=chunk)
                            if result and result["score"] > best_score:
                                best_score = result["score"]
                                best_result = result
                        except Exception as e:
                            continue

                    if best_result and best_result["score"] >= adjusted_threshold:
                        answer = best_result["answer"].strip()
                        if len(answer) >= self.qa_settings.get("min_answer_length", 15):
                            qa_pairs.append({
                                "question": question,
                                "answer": answer,
                                "source": "keyword_generated",
                                "confidence": round(best_result["score"], 2)
                            })
            except ImportError:
                logger.warning("NLTK not available for keyword extraction")
            except Exception as e:
                logger.debug(f"Keyword extraction failed: {str(e)}")

            return qa_pairs
        except Exception as e:
            logger.error(f"QA generation failed: {str(e)}")
            return []

    def _force_generate_qa(self, text: str) -> List[Dict]:
        """Force generate QA pairs even when no explicit ones found."""
        if not self.model or not text:
            logger.warning("Model not loaded or text is empty.")
            return []

        try:
            logger.info("Entering force generation mode...")

            # Break text into manageable chunks for processing
            chunks = []
            chunk_size = 1500
            for i in range(0, min(50000, len(text)), chunk_size):
                chunk = text[i:i + chunk_size * 2]
                if len(chunk) > 300:
                    chunks.append(chunk)

            if not chunks:
                logger.warning("No valid text chunks found for generation.")
                return []

            logger.info(f"Processing {len(chunks)} chunks for forced generation.")

            # Curated list of questions that typically yield good results
            direct_questions = [
                "What is the main topic discussed in this document?",
                "What are the key concepts explained in this text?",
                "What is the most important information in this document?",
                "What are the main benefits described in this document?",
                "What is the purpose of this document?",
                "What problem does this document address?",
                "What are the key takeaways from this document?",
                "How would you summarize this document?",
                "What are the main points covered in this document?",
                "What is the conclusion of this document?"
            ]

            qa_pairs = []
            confidence_threshold = self.qa_settings.get("confidence_threshold", 0.6)
            adjusted_threshold = max(0.3, confidence_threshold - 0.2)

            for question in direct_questions[:5]:
                best_answer = None
                best_score = 0.0

                for chunk in chunks[:3]:
                    try:
                        result = self.model(question=question, context=chunk)
                        if result and result["score"] > best_score:
                            best_score = result["score"]
                            best_answer = result
                    except Exception as e:
                        logger.debug(f"Error processing chunk for question '{question}': {str(e)}")
                        continue

                if best_answer and best_score >= adjusted_threshold:
                    answer = best_answer["answer"].strip()
                    if len(answer) >= self.min_answer_length:
                        qa_pairs.append({
                            "question": question,
                            "answer": answer,
                            "source": "forced_generated",
                            "confidence": round(best_score, 2)
                        })

            logger.info(f"Force-generated {len(qa_pairs)} QA pairs.")
            return qa_pairs

        except Exception as e:
            logger.error(f"Force QA generation failed: {str(e)}")
            return []

    def _create_question_for_content(self, sentence: str) -> str:
        """Create a reasonable question for a given sentence"""
        # Simple rules-based approach for creating questions
        sentence = sentence.strip()
        # Look for named entities or key phrases
        if re.search(r'(is|are|was|were)\s+([^.?!]+)', sentence, re.IGNORECASE):
            # "X is Y" -> "What is X?"
            match = re.search(r'([^.?!,]+)\s+(is|are|was|were)\s+([^.?!]+)', sentence, re.IGNORECASE)
            if match:
                subject = match.group(1).strip()
                return f"What {match.group(2)} {subject}?"

        # Fall back to generic questions
        if "how to" in sentence.lower() or "steps" in sentence.lower():
            return "How would you accomplish this task?"
        elif any(word in sentence.lower() for word in ["advantage", "benefit", "pro", "con"]):
            return "What are the advantages and disadvantages?"
        elif any(word in sentence.lower() for word in ["define", "definition", "mean"]):
            match = re.search(r'(define|definition|mean)s?\s+of\s+([^.?!,]+)', sentence, re.IGNORECASE)
            if match:
                term = match.group(2).strip()
                return f"What is {term}?"

        # Default fallback
        return f"Can you explain more about this: '{sentence[:50]}...'?"

    def process(self, text: str) -> List[Dict]:
        """Universal Q&A processing"""
        if self.mode == "extract":
            qa_pairs = self._extract_explicit_qa(text)
            logger.info(f"Extracted {len(qa_pairs)} explicit Q&A pairs")
            return qa_pairs

        elif self.mode == "generate":
            qa_pairs = self._generate_qa_from_content(text)
            logger.info(f"Generated {len(qa_pairs)} Q&A pairs")
            return qa_pairs

        else:  # auto mode - try extraction first, then generation
            qa_pairs = self._extract_explicit_qa(text)
            if qa_pairs:
                logger.info(f"Auto mode: Extracted {len(qa_pairs)} explicit Q&A pairs")
                return qa_pairs

            qa_pairs = self._generate_qa_from_content(text)
            logger.info(f"Auto mode: Generated {len(qa_pairs)} Q&A pairs after extraction found none")
            return qa_pairs