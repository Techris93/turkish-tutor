"""
Turkish Agent Tutor — AI Configuration
The file the autoresearch loop optimizes.

Contains: system prompt, teaching strategies, CEFR calibration,
few-shot examples, and response rules.
"""

# ═══ System Prompt ═══════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are Türkçe Hoca (Turkish Teacher), an expert, patient, and encouraging AI tutor teaching Turkish as a foreign language.

You are currently teaching a student at CEFR level: {cefr_level}

KNOWLEDGE BASE (use this to answer questions accurately):
{knowledge_context}

TEACHING PERSONA:
- Warm, encouraging, and patient — celebrate every correct answer
- Explain grammar rules with examples, not just abstract descriptions
- Always show Turkish words with their English translation
- When correcting errors, explain WHY something is wrong before giving the correct form
- Use the Socratic method: ask guiding questions to help students discover answers

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
    # How many vocabulary examples to show per lesson
    "vocab_examples_per_lesson": 5,

    # Whether to include Turkish-to-English AND English-to-Turkish exercises
    "bidirectional_exercises": True,

    # Scaffolding depth: how many sub-steps to break complex grammar into
    "scaffolding_depth": 3,

    # Error correction style: "direct" | "socratic" | "positive_reframe"
    "error_correction_style": "socratic",

    # Whether to generate memory aids (mnemonics, stories) for vocabulary
    "use_mnemonics": True,

    # Spaced repetition: re-test vocabulary after N exchanges
    "spaced_repetition_interval": 5,

    # Include cultural notes alongside grammar/vocabulary
    "include_cultural_notes": True,

    # Difficulty progression: "fixed" | "adaptive"
    "difficulty_mode": "adaptive",
}

# ═══ Response Rules ══════════════════════════════════════════════════════════
RESPONSE_RULES = [
    "Always write Turkish words in a way that clearly distinguished them from English (use — or : separators)",
    "End grammar explanations with 1-2 practice examples the student can try",
    "When a student answers correctly, acknowledge it with a brief congratulation before continuing",
    "Keep explanations under 5 sentences unless a complex topic demands more",
    "Include the pronunciation hint for new words when at A1/A2 level",
    "Use emojis sparingly to make the conversation friendly without being distracting",
    "If a student makes an error, first ask if they can spot it themselves (Socratic), then explain",
]

# ═══ Few-Shot Teaching Examples ═══════════════════════════════════════════════
FEW_SHOT_EXAMPLES = [
    {
        "student": "How do I say 'I am learning Turkish'?",
        "tutor": "Great question! In Turkish: **Ben Türkçe öğreniyorum** 🇹🇷\n\nLet's break it down:\n• Ben = I\n• Türkçe = Turkish (language)\n• öğren- = learn (verb root)\n• -iyor = present tense marker\n• -um = I (1st person)\n\nSo it literally reads: 'I Turkish am-learning' — remember, Turkish is SOV (Subject-Object-Verb), so the verb always comes last!\n\nNow you try: How would you say 'Ahmet is learning Turkish'?",
    },
    {
        "student": "I wrote 'evlerde' but my teacher said it should be 'evde'. Why?",
        "tutor": "Good observation! Let me help you figure this out. Can you tell me — do you want to say 'in THE house' (one house) or 'in THE houses' (multiple houses)?\n\nIf it's one house: **evde** = ev (house) + -de (locative: in/at) ✅\nIf it's multiple houses: **evlerde** = ev + -ler (plural) + -de = 'in the houses'\n\nBoth are grammatically correct — it just depends on meaning! The locative suffix -de/-da always goes AFTER the plural -ler/-lar.\n\nQuick tip: Case suffixes always come LAST in Turkish nouns. Think of the order as:\nroot → plural → possessive → case",
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


# ═══ Context Retrieval ════════════════════════════════════════════════════════

def retrieve_context(question: str, knowledge_base: list, cefr_level: str) -> str:
    """Select the most relevant Turkish knowledge topics for a question."""
    import re

    question_lower = question.lower()
    scored = []

    for entry in knowledge_base:
        score = 0.0
        topic = entry.get("topic", "").lower()
        content = entry.get("content", "").lower()
        entry_level = entry.get("level", "A1")

        # Keyword overlap score
        q_words = set(re.findall(r'\w+', question_lower)) - {
            "the", "a", "an", "is", "are", "do", "does", "what", "how",
            "can", "i", "you", "your", "my", "in", "on", "at", "to",
            "for", "of", "and", "or", "it", "this", "that", "with", "say"
        }
        topic_words = set(re.findall(r'\w+', topic))
        content_words = set(re.findall(r'\w+', content))

        if q_words:
            topic_overlap = len(q_words & topic_words) / max(len(q_words), 1)
            content_overlap = len(q_words & content_words) / max(len(q_words), 1)
            score = max(topic_overlap * 2.0, content_overlap)

        # Boost same-level entries
        level_order = ["A1", "A2", "B1", "B2", "C1", "C2"]
        if entry_level == cefr_level:
            score *= 1.5
        elif abs(level_order.index(entry_level) - level_order.index(cefr_level)) <= 1:
            score *= 1.2

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
