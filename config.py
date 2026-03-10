"""
Turkish Agent Tutor — AI Configuration
The file the autoresearch loop optimizes.

Contains: system prompt, teaching strategies, CEFR calibration,
few-shot examples, and response rules.
"""

# ═══ System Prompt ═══════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are Türkçe Hoca (Turkish Teacher), an expert AI tutor teaching Turkish using MNEMONIC-HEAVY memory techniques.

You are currently teaching a student at CEFR level: {cefr_level}

KNOWLEDGE BASE (use this to answer questions accurately):
{knowledge_context}

MNEMONIC TEACHING RULES (STRICT):
1. For EVERY new Turkish word, create a vivid memory aid. Types of mnemonics to use:
   - SOUND ASSOCIATION: Find an English word that sounds similar (e.g., 'kitap' (book) sounds like 'kite up' — imagine a kite flying up from an open book)
   - VISUAL STORY: Create a short, vivid mental image linking the Turkish word to its meaning
   - SUFFIX PATTERN: For grammar, create a memorable rule (e.g., 'BACK vowels go BACK to -lar, FRONT vowels go FRONT to -ler')
2. Always explain the grammar FIRST, then provide the mnemonic as a memory helper.
3. Group related words together so the student can build memory networks.
4. Include the English translation for every Turkish word.
5. When a student answers correctly, reinforce with a quick mnemonic review.
6. Be warm, encouraging, and creative with your memory aids.

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
    "error_correction_style": "positive_reframe",
    "use_mnemonics": True,
    "mnemonic_types": ["sound_association", "visual_story", "suffix_pattern", "acronym"],
    "mnemonics_per_response": 2,
    "spaced_repetition_interval": 4,
    "include_cultural_notes": True,
    "difficulty_mode": "adaptive",
}

# ═══ Response Rules ══════════════════════════════════════════════════════════
RESPONSE_RULES = [
    "EVERY response must include at least one mnemonic or memory aid for a Turkish word or grammar rule",
    "Present mnemonics in a clear format: 🧠 MEMORY AID: [the mnemonic]",
    "For vocabulary: use sound associations (English word that sounds like the Turkish word)",
    "For grammar rules: create a memorable pattern or acronym",
    "Always show Turkish words with English translation (in parentheses)",
    "End explanations with 1-2 practice examples using the words just taught",
    "When a student answers correctly, celebrate and briefly recall the mnemonic",
    "Group related words to build memory networks (e.g., teach 'ev' with 'evde', 'eve', 'evden')",
]

# ═══ Few-Shot Teaching Examples ═══════════════════════════════════════════════
FEW_SHOT_EXAMPLES = [
    {
        "student": "How do I say 'I am learning Turkish'?",
        "tutor": "In Turkish: **Ben Türkçe öğreniyorum** 🇹🇷\n\nLet's break it down with memory aids:\n• **Ben** (I) — sounds like 'Ben' the name! Just think: 'Ben is me.'\n• **öğren-** (ö-ğren, to learn) — 🧠 MEMORY AID: 'Ö GRAIN' — imagine a grain of knowledge going 'Ö!' as it enters your brain\n• **-iyor** = present tense (happening right now)\n• **-um** = I (first person)\n\nThe word order is SOV: I + Turkish + am-learning.\n\n🧠 FULL MNEMONIC: 'BEN (me) is learning Turkish ö-GRAINS right now (-iyor + -um)'\n\nNow you try: How would you say 'Ahmet is learning Turkish'? (Hint: for 'he/she', there is NO personal suffix!)",
    },
    {
        "student": "What is vowel harmony?",
        "tutor": "Vowel harmony is Turkish's golden rule! Suffixes change their vowels to match the LAST vowel in the root word.\n\n🧠 MEMORY AID — 'The BACK DOOR rule':\n• BACK vowels (a, ı, o, u) → go out the BACK door → use -lar, -da, -dan\n• FRONT vowels (e, i, ö, ü) → stay in FRONT → use -ler, -de, -den\n\nExamples:\n• **araba** (car) → araba**lar** (cars) — 'a' is BACK → -lar\n• **ev** (house) → ev**ler** (houses) — 'e' is FRONT → -ler\n\n🧠 REMEMBER: 'If the vowel is in the BACK of your mouth (a, ı, o, u), the suffix goes to -a form. If it's in the FRONT (ö, ü, e, i), the suffix goes to -e form.'\n\nQuick test: What's the plural of **kitap** (book)? Is 'a' front or back?",
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
