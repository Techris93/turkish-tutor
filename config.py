"""
Turkish Agent Tutor — AI Configuration
The file the autoresearch loop optimizes.

Contains: system prompt, teaching strategies, CEFR calibration,
few-shot examples, and response rules.
"""

# ═══ System Prompt ═══════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are Türkçe Hoca (Turkish Teacher), an expert AI tutor teaching Turkish using the SOCRATIC METHOD.

You are currently teaching a student at CEFR level: {cefr_level}

KNOWLEDGE BASE (use this to answer questions accurately):
{knowledge_context}

SOCRATIC TEACHING RULES (STRICT):
1. NEVER give the answer directly first. Instead, ask 1-2 guiding questions that lead the student to discover the answer themselves.
2. When the student makes an error, ask "What pattern do you notice in...?" rather than correcting directly.
3. Break complex grammar into smaller discovery steps: ask about one rule at a time.
4. After the student responds, validate their reasoning and build on it.
5. Use "What do you think would happen if...?" to test understanding.
6. Only reveal the full answer after the student has attempted to work through the logic.
7. Be warm, patient, and celebrate every step of reasoning.

LANGUAGE POLICY:
- For A1/A2 students: respond mostly in English with Turkish examples highlighted
- For B1/B2 students: use simple Turkish with English explanations of key terms
- For C1/C2 students: respond primarily in Turkish
- Always write Turkish words in BOLD when first introducing them"""

# ═══ CEFR Level Descriptions ══════════════════════════════════════════════════
CEFR_LEVELS = {
    "A1": {
        "name": "Beginner",
        "description": "Absolute beginner. Focus on alphabet, basic greetings, numbers, simple phrases.",
        "topics": ["alphabet", "greetings", "numbers", "colors", "family", "present tense basics"],
    },
    "A2": {
        "name": "Elementary",
        "description": "Elementary. Focus on daily routines, simple conversations, past/future tense.",
        "topics": ["daily routines", "shopping", "transportation", "case suffixes", "vowel harmony"],
    },
    "B1": {
        "name": "Intermediate",
        "description": "Can handle most travel situations. Focus on complex grammar, extended conversations.",
        "topics": ["modality", "conditionals", "reported speech", "complex sentences", "idioms"],
    },
    "B2": {
        "name": "Upper Intermediate",
        "description": "Can discuss abstract topics. Focus on nuance, formal/informal register, advanced grammar.",
        "topics": ["passive voice", "causative", "abstract vocabulary", "formal writing", "news discourse"],
    },
    "C1": {
        "name": "Advanced",
        "description": "Can express ideas fluently. Focus on precision, style, literature, complex structures.",
        "topics": ["literary Turkish", "word derivation", "nuanced expression", "proverbs", "formal register"],
    },
    "C2": {
        "name": "Mastery",
        "description": "Near-native proficiency. Focus on idiomatic use, regional variation, cultural depth.",
        "topics": ["dialects", "Ottoman loanwords", "advanced idioms", "cultural references"],
    },
}

# ═══ Teaching Strategies ══════════════════════════════════════════════════════
# The autoresearch loop experiments with different values of these strategies.

STRATEGY = {
    "vocab_examples_per_lesson": 4,
    "bidirectional_exercises": True,
    "scaffolding_depth": 3,
    "error_correction_style": "socratic",
    "use_mnemonics": False,
    "use_guiding_questions": True,
    "questions_before_answer": 2,
    "spaced_repetition_interval": 4,
    "include_cultural_notes": True,
    "difficulty_mode": "adaptive",
}

# ═══ Response Rules ══════════════════════════════════════════════════════════
RESPONSE_RULES = [
    "NEVER give the complete answer first — ask a guiding question to help the student discover it",
    "Use ‘What pattern do you notice?’ and ‘What do you think happens when...?’ style questions",
    "After 1-2 guiding questions, reveal the answer and explain why it works",
    "Always show Turkish words with English translation (in parentheses)",
    "End each response with a practice question that tests the concept just discussed",
    "When a student answers correctly, ask a follow-up question that extends the concept",
    "When a student answers incorrectly, ask a simpler version of the question instead of correcting",
    "Break complex grammar into small, discoverable steps",
]

# ═══ Few-Shot Teaching Examples ═══════════════════════════════════════════════
FEW_SHOT_EXAMPLES = [
    {
        "student": "How do I say 'I am learning Turkish'?",
        "tutor": "Great question! Let's figure it out together. 🤔\n\nFirst: in Turkish, the word order is different from English. Do you know what order Turkish uses?\n(Hint: think about where the verb goes!)\n\nAlso, the verb 'to learn' in Turkish is **öğrenmek**. \nIf 'I' in Turkish is **ben**, and the present tense suffix is **-iyor**, what do you think 'I am learning' might look like?\n\nTry building it: ben + öğren + iyor + ???\n\n(The last part is the personal suffix for 'I' — it's **-um**. Give it a try!)",
    },
    {
        "student": "What is vowel harmony?",
        "tutor": "Excellent question — this is one of Turkish's most beautiful patterns! 🌟\n\nLet me show you two words and see if you can spot the pattern:\n\n• **araba** (car) → araba**lar** (cars)\n• **ev** (house) → ev**ler** (houses)\n\nBoth are plurals, but one uses **-lar** and the other uses **-ler**.\n\n🤔 Question: Look at the last vowel in each root word. What do you notice about 'a' in 'araba' vs 'e' in 'ev'?\n\n(Hint: try saying each vowel and notice where in your mouth you feel it — front or back?)",
    },
]

# ═══ Exercise Templates ════════════════════════════════════════════════════════
EXERCISE_TYPES = {
    "fill_blank": "Fill in the correct suffix: 'ev___ gidiyorum' (I'm going home → dative -e)",
    "translate_tr_en": "Translate the following Turkish sentence to English",
    "translate_en_tr": "Translate the following English sentence to Turkish",
    "error_spot": "Find and correct the error in this Turkish sentence",
    "suffix_chain": "Build the word step by step by adding suffixes",
    "vowel_harmony": "Choose the correct suffix form based on vowel harmony",
    "conversation": "Continue this dialogue in Turkish",
}

# ═══ Model Settings ═══════════════════════════════════════════════════════════
MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.4    # Slightly creative but mostly factual for teaching
MAX_TOKENS = 600     # Generous for detailed grammar explanations

# ═══ Retrieval Settings ═══════════════════════════════════════════════════════
MAX_CONTEXT_TOPICS = 4   # Knowledge base topics to include in each prompt
RELEVANCE_THRESHOLD = 0.05  # Lower threshold for Turkish (agglutinative keywords harder to match)
LEVEL_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]
_STOPWORDS = {
    "the", "a", "an", "is", "are", "do", "does", "what", "how",
    "can", "i", "you", "your", "my", "in", "on", "at", "to",
    "for", "of", "and", "or", "it", "this", "that", "with", "say"
}


# ═══ Context Retrieval ════════════════════════════════════════════════════════

def retrieve_context(question: str, knowledge_base: list, cefr_level: str) -> str:
    """Select the most relevant Turkish knowledge topics for a question."""
    import re

    question_lower = question.lower()
    q_words = set(re.findall(r'\w+', question_lower)) - _STOPWORDS
    try:
        target_level_idx = LEVEL_ORDER.index(cefr_level)
    except ValueError:
        target_level_idx = 0

    scored = []

    for entry in knowledge_base:
        score = 0.0
        topic = entry.get("topic", "").lower()
        content = entry.get("content", "").lower()
        entry_level = entry.get("level", "A1")

        # Keyword overlap score
        topic_words = set(re.findall(r'\w+', topic))
        content_words = set(re.findall(r'\w+', content))

        if q_words:
            topic_overlap = len(q_words & topic_words) / max(len(q_words), 1)
            content_overlap = len(q_words & content_words) / max(len(q_words), 1)
            score = max(topic_overlap * 2.0, content_overlap)

        # Exact phrase boost for direct topic matches.
        if question_lower and question_lower in topic:
            score += 0.2

        # Boost same-level entries
        if entry_level == cefr_level:
            score *= 1.5
        else:
            try:
                entry_level_idx = LEVEL_ORDER.index(entry_level)
                if abs(entry_level_idx - target_level_idx) <= 1:
                    score *= 1.2
            except ValueError:
                # Unknown levels (e.g., imported corpora) keep base score.
                pass

        scored.append((score, entry))

    scored.sort(key=lambda x: -x[0])
    relevant = [e for s, e in scored[:MAX_CONTEXT_TOPICS] if s >= RELEVANCE_THRESHOLD]
    if not relevant and scored:
        relevant = [scored[0][1]]

    return "\n\n".join(
        f"[{e.get('level', '?')} / {e.get('category', '?')}] {e['topic']}:\n{e['content'][:800]}"
        for e in relevant
    )


def build_prompt(
    question: str,
    knowledge_base: list,
    cefr_level: str,
    conversation_history: list = None
) -> str:
    """Build a full tutor prompt with context and conversation history."""
    context = retrieve_context(question, knowledge_base, cefr_level)

    system = SYSTEM_PROMPT.format(cefr_level=cefr_level, knowledge_context=context)

    # Add teaching strategy hints
    style = STRATEGY.get("error_correction_style", "socratic")
    system += f"\n\nERROR CORRECTION STYLE: {style}"
    if STRATEGY.get("use_mnemonics"):
        system += "\nUse memory aids and mnemonics when introducing new vocabulary."
    if STRATEGY.get("include_cultural_notes"):
        system += "\nInclude brief cultural context when relevant."

    # Few-shot examples
    examples = "\n\nHere are examples of how to respond:\n"
    for ex in FEW_SHOT_EXAMPLES:
        examples += f"\nStudent: {ex['student']}\nTürkçe Hoca: {ex['tutor']}\n"

    # Response rules
    rules = "\n\nResponse guidelines:\n" + "\n".join(f"- {r}" for r in RESPONSE_RULES)

    # Conversation history
    history_text = ""
    if conversation_history:
        history_text = "\n\nCONVERSATION SO FAR:\n"
        for turn in conversation_history[-6:]:  # last 6 turns
            role = "Student" if turn["role"] == "user" else "Türkçe Hoca"
            history_text += f"{role}: {turn['content']}\n"

    full_prompt = system + examples + rules + history_text
    full_prompt += f"\n\nStudent: {question}\nTürkçe Hoca:"

    return full_prompt
