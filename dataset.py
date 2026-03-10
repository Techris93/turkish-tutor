"""
Turkish Agent Tutor — Dataset Pipeline
Builds the knowledge base from curated Turkish language data.

Sources:
  1. Built-in curated dataset (vocabulary, grammar rules, dialogues)
  2. Optional: HuggingFace datasets (TQuAD, OPUS-TR)
  3. Optional: Live scraping of TDK (Turkish Language Association)

Usage:
    python dataset.py                        # Build full knowledge base
    python dataset.py --fetch-hf             # Also fetch from HuggingFace
    python dataset.py --status               # Show knowledge base stats
"""

import json
import os
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Any

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
KNOWLEDGE_FILE = os.path.join(DATA_DIR, "knowledge.json")
TEST_FILE = os.path.join(DATA_DIR, "test_qa.json")


# ═══ Built-In Curated Turkish Knowledge Base ══════════════════════════════════

VOCABULARY = {
    "A1": {
        "Greetings & Basics": [
            ("Merhaba", "Hello"),
            ("Günaydın", "Good morning"),
            ("İyi günler", "Good day"),
            ("İyi akşamlar", "Good evening"),
            ("İyi geceler", "Good night"),
            ("Hoşça kal", "Goodbye (said to the one leaving)"),
            ("Güle güle", "Goodbye (said to the one staying)"),
            ("Teşekkür ederim", "Thank you"),
            ("Rica ederim", "You're welcome"),
            ("Lütfen", "Please"),
            ("Evet", "Yes"),
            ("Hayır", "No"),
            ("Belki", "Maybe"),
            ("Tamam", "OK / Alright"),
            ("Pardon / Özür dilerim", "Excuse me / Sorry"),
        ],
        "Numbers 1–20": [
            ("bir", "1"), ("iki", "2"), ("üç", "3"), ("dört", "4"), ("beş", "5"),
            ("altı", "6"), ("yedi", "7"), ("sekiz", "8"), ("dokuz", "9"), ("on", "10"),
            ("on bir", "11"), ("on iki", "12"), ("on üç", "13"), ("on dört", "14"),
            ("on beş", "15"), ("on altı", "16"), ("on yedi", "17"), ("on sekiz", "18"),
            ("on dokuz", "19"), ("yirmi", "20"),
        ],
        "Colors": [
            ("kırmızı", "red"), ("mavi", "blue"), ("yeşil", "green"),
            ("sarı", "yellow"), ("siyah", "black"), ("beyaz", "white"),
            ("turuncu", "orange"), ("mor", "purple"), ("pembe", "pink"),
            ("kahverengi", "brown"), ("gri", "grey"),
        ],
        "Family": [
            ("anne", "mother"), ("baba", "father"), ("kız kardeş", "sister"),
            ("erkek kardeş", "brother"), ("büyükanne", "grandmother"),
            ("büyükbaba", "grandfather"), ("amca", "uncle (paternal)"),
            ("teyze", "aunt (maternal)"), ("kuzen", "cousin"), ("eş / karı / koca", "spouse"),
        ],
        "Days of the Week": [
            ("Pazartesi", "Monday"), ("Salı", "Tuesday"), ("Çarşamba", "Wednesday"),
            ("Perşembe", "Thursday"), ("Cuma", "Friday"), ("Cumartesi", "Saturday"),
            ("Pazar", "Sunday"),
        ],
        "Common Verbs": [
            ("gitmek", "to go"), ("gelmek", "to come"), ("yapmak", "to do / make"),
            ("yemek", "to eat"), ("içmek", "to drink"), ("uyumak", "to sleep"),
            ("konuşmak", "to speak"), ("anlamak", "to understand"),
            ("bilmek", "to know"), ("sevmek", "to love / like"),
            ("istemek", "to want"), ("görmek", "to see"),
        ],
    },
    "A2": {
        "Time Expressions": [
            ("şimdi", "now"), ("bugün", "today"), ("yarın", "tomorrow"),
            ("dün", "yesterday"), ("bu hafta", "this week"), ("geçen hafta", "last week"),
            ("gelecek hafta", "next week"), ("sabah", "morning"), ("öğle", "noon"),
            ("akşam", "evening"), ("gece", "night"),
        ],
        "Transportation": [
            ("otobüs", "bus"), ("metro", "metro / subway"), ("taksi", "taxi"),
            ("uçak", "airplane"), ("tren", "train"), ("vapur", "ferry"),
            ("araba", "car"), ("bisiklet", "bicycle"), ("yürümek", "to walk"),
        ],
        "Shopping": [
            ("market", "supermarket"), ("fiyat", "price"), ("kaç lira?", "how much? (in lira)"),
            ("ucuz", "cheap"), ("pahalı", "expensive"), ("indirim", "discount"),
            ("kasa", "checkout / cashier"), ("fatura / fiş", "receipt"),
        ],
        "Weather": [
            ("hava", "weather / air"), ("güneşli", "sunny"), ("yağmurlu", "rainy"),
            ("karlı", "snowy"), ("bulutlu", "cloudy"), ("sıcak", "hot"),
            ("soğuk", "cold"), ("rüzgarlı", "windy"), ("sıcaklık", "temperature"),
        ],
    },
    "B1": {
        "Emotions": [
            ("mutlu", "happy"), ("üzgün", "sad"), ("kızgın", "angry"),
            ("endişeli", "worried / anxious"), ("şaşkın", "surprised"),
            ("yorgun", "tired"), ("heyecanlı", "excited"), ("korkmuş", "scared"),
            ("memnun", "pleased / satisfied"),
        ],
        "Work & Career": [
            ("iş", "job / work"), ("ofis", "office"), ("toplantı", "meeting"),
            ("maaş", "salary"), ("terfi", "promotion"), ("istifa etmek", "to resign"),
            ("işe alınmak", "to be hired"), ("kariyer", "career"),
            ("iş arkadaşı", "colleague"), ("yönetici / müdür", "manager"),
        ],
        "Health": [
            ("hasta", "sick"), ("doktor", "doctor"), ("hastane", "hospital"),
            ("ilaç", "medicine"), ("ağrı", "pain / ache"), ("ateş", "fever"),
            ("ameliyat", "surgery"), ("sigorta", "insurance"), ("eczane", "pharmacy"),
        ],
    },
    "B2": {
        "Abstract Concepts": [
            ("özgürlük", "freedom"), ("adalet", "justice"), ("eşitlik", "equality"),
            ("demokrasi", "democracy"), ("sorumluluk", "responsibility"),
            ("güven", "trust / confidence"), ("başarı", "success"),
            ("hayal kırıklığı", "disappointment"), ("fırsat", "opportunity"),
        ],
        "Academic Verbs": [
            ("analiz etmek", "to analyze"), ("tartışmak", "to discuss / argue"),
            ("karşılaştırmak", "to compare"), ("değerlendirmek", "to evaluate"),
            ("araştırmak", "to research"), ("incelemek", "to examine"),
            ("önermek", "to propose / suggest"),
        ],
    },
}


GRAMMAR_RULES = [
    {
        "topic": "Vowel Harmony — The Foundation of Turkish",
        "level": "A1",
        "content": """Turkish vowel harmony is the most important rule to learn. Suffixes change their vowels to match the LAST vowel of the root word.

TWO-WAY HARMONY (e/a): Suffixes like the plural -ler/-lar, locative -de/-da, ablative -den/-dan follow this rule:
  • If the last vowel is a FRONT vowel (e, i, ö, ü) → use 'e' form: -ler, -de, -den
  • If the last vowel is a BACK vowel (a, ı, o, u) → use 'a' form: -lar, -da, -dan

Examples:
  ev (house) → evler (houses) — 'e' is front → -ler
  araba (car) → arabalar (cars) — 'a' is back → -lar
  okul (school) → okulda (at school) — 'u' is back → -da
  park (park) → parktan (from the park) — 'a' is back → -tan

FOUR-WAY HARMONY (i/ı/ü/u): Used in suffixes like the accusative -i/-ı/-ü/-u:
  • e, i → -i  |  a, ı → -ı  |  ö, ü → -ü  |  o, u → -u

Examples:
  kitap (book) → kitabı (the book, accusative) — 'a' → -ı
  ev (house) → evi (the house, accusative) — 'e' → -i
  göz (eye) → gözü (the eye, accusative) — 'ö' → -ü
  oda (room) → odayı (the room, accusative) — 'a' → -ı

FRONT vowels: e, i, ö, ü
BACK vowels: a, ı, o, u""",
    },
    {
        "topic": "Agglutination — Building Words with Suffixes",
        "level": "A2",
        "content": """Turkish is agglutinative: you build meaning by stacking suffixes onto root words. Each suffix has ONE clear job.

STRUCTURE: ROOT + suffix1 + suffix2 + suffix3...

Example: 'evlerimizden' (from our houses)
  ev = house
  -ler = plural → evler (houses)
  -imiz = our → evlerimiz (our houses)
  -den = from → evlerimizden (from our houses)

Example: 'kitaplarımda' (in my books)
  kitap = book
  -lar = plural → kitaplar
  -ım = my → kitaplarım
  -da = in/at → kitaplarımda

Example: 'gidebilirdim' (I could have gone)
  git = go (root)
  -ebil = to be able to (ability)
  -ir = aorist (general/habitual)
  -di = past tense
  -m = first person singular
  → gidebilirdim (I could have gone)

The suffixes always follow STRICT ORDER:
  NOUN: root + plural + possessive + case
  VERB: root + voice + negation + tense + person""",
    },
    {
        "topic": "Case Suffixes — The 6 Cases",
        "level": "A2",
        "content": """Turkish has 6 noun cases (suffixes that show the role of a noun in the sentence):

1. NOMINATIVE (subject) — no suffix
   Kedi uyuyor. (The cat is sleeping.) — kedi = subject, no suffix

2. ACCUSATIVE (-i/-ı/-ü/-u) — direct object (specific, definite)
   Kitabı okudum. (I read THE book.) — kitab+ı
   Note: indefinite objects take no suffix: Kitap okudum. (I read A book.)

3. DATIVE (-e/-a) — to, for, direction
   Eve gidiyorum. (I'm going HOME.) — ev+e
   Sana bir şey söyleyeyim. (Let me tell YOU something.) — san+a

4. LOCATIVE (-de/-da) — at, in, on (location)
   Okulda (at school) — okul+da
   Evde (at home) — ev+de

5. ABLATIVE (-den/-dan) — from, out of, about
   Ankara'dan geldim. (I came FROM Ankara.) — Ankara+dan
   Senden korkuyorum. (I'm afraid OF you.) — sen+den

6. GENITIVE (-in/-ın/-ün/-un) — of, 's (possession)
   Türkiye'nin başkenti (the capital OF Turkey) — Türkiye+nin
   Arabanın rengi (the color OF the car) — araba+nın""",
    },
    {
        "topic": "Present Tense — Şimdiki Zaman (-iyor)",
        "level": "A1",
        "content": """The present continuous tense (-iyor) is the most common Turkish tense and covers both 'I am doing' AND 'I do'.

FORMATION: verb root + -iyor + personal suffix
The -iyor follows 4-way vowel harmony in spoken Turkish but is always spelled -iyor (exception to the rule).

Personal suffixes:
  ben (I) → -um/-üm/-ım/-im: gidiyorum (I am going)
  sen (you) → -sun/-sün/-sın/-sin: gidiyorsun (you are going)
  o (he/she/it) → (no suffix): gidiyor (he/she is going)
  biz (we) → -uz/-üz/-ız/-iz: gidiyoruz (we are going)
  siz (you plural/formal) → -sunuz/-sünüz/-sınız/-siniz: gidiyorsunuz
  onlar (they) → -lar/-ler: gidiyorlar (they are going)

VERB ROOT RULES:
  • If verb root ends in a vowel, drop it before -iyor: gel→ gel+iyor, ye → y+iyor → yiyor
  • If root ends in consonant, add buffer vowel (harmony): git → gid+iyor → gidiyor

NEGATION: insert -mi- before -iyor: gitmiyorum (I am not going)
QUESTION: add mi/mı/mü/mu after tense: Gidiyor musun? (Are you going?)

Examples:
  Türkçe öğreniyorum. (I am learning Turkish.)
  Ne yapıyorsun? (What are you doing?)
  Yağmur yağıyor. (It is raining.)""",
    },
    {
        "topic": "Past Tense — Geçmiş Zaman (-di)",
        "level": "A2",
        "content": """Turkish has TWO past tenses. The -di past (witnessed/direct experience) is the most common.

FORMATION: verb root + -di/-dı/-dü/-du (or -ti/-tı/-tü/-tu after voiceless consonants) + personal suffix

Voiceless consonants (use -ti form): p, ç, t, k, f, h, s, ş
Voiced or vowel (use -di form): all others

Personal suffixes:
  ben → -m: gittim (I went)
  sen → -n: gittin (you went)
  o → (none): gitti (he/she went)
  biz → -k: gittik (we went)
  siz → -niz/-nız/-nüz/-nuz: gittiniz (you went)
  onlar → -ler/-lar: gittiler (they went)

Examples:
  Dün İstanbul'a gittim. (I went to Istanbul yesterday.)
  Ne zaman geldin? (When did you come?)
  Filmi izledik. (We watched the film.)

THE -MIŞ PAST (reported/hearsay — not witnessed directly):
  Used when you heard about something, not experienced it yourself.
  Uyumuş. (Apparently he slept. / I heard he slept.)
  Gelmiş! (Apparently he came! — used for surprises too)""",
    },
    {
        "topic": "Future Tense — Gelecek Zaman (-ecek/-acak)",
        "level": "A2",
        "content": """FORMATION: verb root + -ecek/-acak + personal suffix (follow 2-way harmony: e→ecek, a→acak)

Personal suffixes:
  ben → -im/-ım: gideceğim (I will go)
  sen → -sin/-sın: gideceksin (you will go)
  o → (none): gidecek (he/she will go)
  biz → -iz/-ız: gideceğiz (we will go)
  siz → -siniz/-sınız: gideceksiniz (you will go)
  onlar → -ler/-lar: gidecekler (they will go)

Note: -ecek → -eceğ- before vowel suffixes (consonant softening: k→ğ)

NEGATION: -me/-ma + yacak/-yecek: gitmeyeceğim (I will not go)

Examples:
  Yarın seni arayacağım. (I will call you tomorrow.)
  Bu yaz Türkiye'ye gideceğiz. (We will go to Turkey this summer.)
  Ne zaman geleceksin? (When will you come?)""",
    },
    {
        "topic": "Question Formation",
        "level": "A1",
        "content": """Turkish questions use a QUESTION PARTICLE that follows vowel harmony: mi / mı / mü / mu

RULE: Write the question particle SEPARATELY from the word it follows.
  • After e, i → mi
  • After a, ı → mı
  • After ö, ü → mü
  • After o, u → mu

YES/NO QUESTIONS: Place mi/mı after the verb (before personal suffix):
  Geliyor musun? (Are you coming?) — geliyor + mu + sun
  Türkçe biliyor musunuz? (Do you know Turkish?)
  Evde misin? (Are you at home?) — after 'i' → mi

WH-QUESTIONS use question words (no mi needed):
  Ne? (What?), Nerede? (Where?), Nereye? (Where to?), Ne zaman? (When?)
  Nasıl? (How?), Neden/Niçin? (Why?), Kim? (Who?), Kaç? (How many?)
  Hangi? (Which?)

WORD ORDER: In Turkish, the question word takes the same position as its answer:
  Saat kaç? (What time is it? — lit. 'Clock how-many?')
  Kim geldi? (Who came?)
  Neret gittin? (Where did you go?)""",
    },
    {
        "topic": "Possessive Suffixes",
        "level": "A2",
        "content": """Possession in Turkish uses suffixes, not separate words like 'my', 'your', 'his'.

POSSESSIVE SUFFIXES (attach to the OWNED noun):
  benim (my) → -im/-ım/-üm/-um: evim (my house), arabam (my car)
  senin (your) → -in/-ın/-ün/-un: evin (your house), araban (your car)
  onun (his/her/its) → -i/-ı/-ü/-u: evi (his/her house), arabası (his/her car)
  bizim (our) → -imiz/-ımız/-ümüz/-umuz: evimiz (our house)
  sizin (your pl.) → -iniz/-ınız/-ünüz/-unuz: eviniz (your house)
  onların (their) → -leri/-ları: evleri (their house)

Buffer 'y' used when root ends in vowel: araba → arabam (not araba+ım)
Buffer 's' for 3rd person when root ends in vowel: araba → arabası

GENITIVE POSSESSION (A's B):
  Ahmet'in evi (Ahmet's house) = Ahmet + -in (genitive) + ev + -i (possessive on owned)
  Türkiye'nin başkenti (Turkey's capital)

Note: The possessor pronoun is optional when clear from context:
  Evim güzel. (My house is beautiful.) — benim is implied""",
    },
    {
        "topic": "Word Order — SOV",
        "level": "A1",
        "content": """Turkish is an SOV (Subject–Object–Verb) language. The VERB always comes LAST.

English: I    love    you.     (SVO — verb in the middle)
Turkish: Ben  seni    seviyorum. (SOV — verb at the end)
         S    O       V

More examples:
  Ben Türkçe öğreniyorum. (I Turkish am-learning. = I am learning Turkish.)
  Ali marketten ekmek aldı. (Ali from-the-market bread bought. = Ali bought bread from the market.)
  Sen ne zaman eve geleceksin? (You when to-home will-come? = When will you come home?)

ADJECTIVES come BEFORE nouns (like English):
  büyük ev (big house) — büyük = big, comes before ev
  güzel Türkçe (beautiful Turkish)

ADVERBS of time usually come at the beginning or after subject:
  Dün markete gittim. (Yesterday I went to the market.)
  Ben her gün Türkçe çalışıyorum. (I every day Turkish study.)

FLEXIBILITY: Turkish word order is flexible because cases show grammatical roles.
  Moving a word earlier emphasizes it (topicalization).
  Seni seviyorum. (I love YOU.) vs. Seviyorum seni. (I love you. — more casual/poetic)""",
    },
    {
        "topic": "Common Suffixes Quick Reference",
        "level": "A2",
        "content": """MOST IMPORTANT SUFFIXES TO MEMORIZE:

NOUN SUFFIXES:
  -ler/-lar = plural: kitap→kitaplar, ev→evler
  -li/-lı/-lü/-lu = with / having: süt+lü = sütlü (with milk); şeker+li = şekerli (with sugar)
  -siz/-sız/-süz/-suz = without: şeker+siz = şekersiz (sugarless); iş+siz = işsiz (unemployed)
  -ci/-cı/-cü/-cu = profession: kitap→kitapçı (bookseller); göz→gözlükçü (optician)
  -lik/-lık/-lük/-luk = abstract noun / container: güzel→güzellik (beauty); şeker→şekerlik (sugar bowl)

VERB SUFFIXES:
  -me/-ma = negation: git+me = gitme (don't go / not going)
  -ebil/-abil = ability (can): git+ebil = gidebil (can go): gidebiliyorum (I can go)
  -meli/-malı = must / should: git+meli = gitmeli (should go): gitmeliyim (I must go)
  -se/-sa = conditional (if): git+se = gitse (if he/she goes): gitsem (if I go)
  -ken = while: okurken (while reading); giderken (while going)

QUESTION:
  -mi/-mı/-mü/-mu = question particle: Geliyor musun? (Are you coming?)""",
    },
    {
        "topic": "Turkish Alphabet & Pronunciation",
        "level": "A1",
        "content": """Turkish uses the Latin alphabet with 29 letters (8 vowels, 21 consonants).

SPECIAL LETTERS not in English:
  Ç/ç = 'ch' sound: çay (tea), çok (very)
  Ğ/ğ = 'soft g' — lengthens the preceding vowel, almost silent: dağ (mountain), yağmur (rain)
  İ/i = normal 'i' sound (with dot — different from I/ı!)
  I/ı = deep 'i' sound, like 'uh': kız (girl), araba (car)
  Ö/ö = like German 'ö' or French 'eu': göz (eye), öğrenci (student)
  Ş/ş = 'sh' sound: şeker (sugar), şimdi (now)
  Ü/ü = like German 'ü' or French 'u': üç (three), gül (rose)

PRONUNCIATION RULES:
  • Every letter is pronounced (no silent letters)
  • Stress is usually on the LAST syllable: ka-PÜ (door), İS-tan-BUL
  • C/c = 'j' sound like 'jam': cam (glass), cep (pocket)
  • V/v = softer than English, like 'w' in some regions: var (there is), vermek (to give)

VOWELS and their sounds:
  a = 'a' in father | e = 'e' in bed | i = 'ee' in see | ı = 'uh' (unstressed)
  o = 'o' in more  | ö = 'ur' in fur | u = 'oo' in food | ü = 'ew' in few""",
    },
    {
        "topic": "Common Turkish Phrases for Daily Life",
        "level": "A1",
        "content": """ESSENTIAL DAILY PHRASES:

Restaurant / Café:
  Menü var mı? — Do you have a menu?
  Ne önerirsiniz? — What do you recommend?
  Hesap lütfen. — The bill/check, please.
  Afiyet olsun! — Enjoy your meal! (Bon appétit)
  Çok lezzetli! — Very delicious!

Directions:
  Nerede? — Where is it?
  Nasıl gidebilirim? — How can I get there?
  Sağa/Sola dönün. — Turn right/left.
  Düz gidin. — Go straight.

Shopping:
  Ne kadar? / Kaç lira? — How much?
  Daha ucuz var mı? — Do you have anything cheaper?
  Bunu alacağım. — I'll take this.
  Kredi kartı kabul ediyor musunuz? — Do you accept credit cards?

Emergency / Help:
  İmdat! — Help!
  Polis çağırın! — Call the police!
  Doktor lazım! — I need a doctor!
  Kayboldum. — I got lost.

Polite Expressions:
  Geçmiş olsun. — Get well soon. (lit. 'May it pass')
  Kolay gelsin. — May it come easy. (said to someone working)
  Başarılar! — Good luck! (lit. 'Successes!')
  İnşallah! — If God wills it / Hopefully
  Maşallah! — How wonderful! (expression of admiration)""",
    },
]


# ═══ Test Q&A Dataset ══════════════════════════════════════════════════════════

TEST_QA = [
    # Vowel Harmony
    {"question": "What suffix makes 'araba' (car) plural?", "expected_answer": "arabalar (-lar because 'a' is a back vowel)", "category": "vowel_harmony", "level": "A1", "difficulty": "easy"},
    {"question": "What suffix makes 'ev' (house) plural?", "expected_answer": "evler (-ler because 'e' is a front vowel)", "category": "vowel_harmony", "level": "A1", "difficulty": "easy"},
    {"question": "How do you say 'at school' in Turkish?", "expected_answer": "okulda (okul + -da, back vowel 'u' → -da)", "category": "case_suffixes", "level": "A2", "difficulty": "easy"},
    {"question": "How do you say 'I am coming' in Turkish?", "expected_answer": "Geliyorum (gel + iyor + um)", "category": "present_tense", "level": "A1", "difficulty": "easy"},
    {"question": "What is the past tense of 'gitmek' (to go) for 'I'?", "expected_answer": "Gittim (git + ti + m)", "category": "past_tense", "level": "A2", "difficulty": "medium"},
    {"question": "How do you form a yes/no question in Turkish?", "expected_answer": "Add the question particle mi/mı/mü/mu (matching vowel harmony) after the verb or predicate, written separately.", "category": "questions", "level": "A1", "difficulty": "medium"},
    {"question": "What word order does Turkish use?", "expected_answer": "SOV — Subject, Object, Verb. The verb always comes at the end.", "category": "word_order", "level": "A1", "difficulty": "easy"},
    {"question": "How do you say 'my house' in Turkish?", "expected_answer": "evim (ev + -im, possessive suffix for 'my')", "category": "possession", "level": "A2", "difficulty": "easy"},
    {"question": "What does the suffix -li/-lı mean?", "expected_answer": "It means 'with' or 'having': sütlü = with milk, şekerli = with sugar", "category": "suffixes", "level": "A2", "difficulty": "medium"},
    {"question": "How do you say 'I will go' in Turkish?", "expected_answer": "Gideceğim (git + ecek + im, future tense)", "category": "future_tense", "level": "A2", "difficulty": "medium"},
    {"question": "What is the difference between -di past and -miş past?", "expected_answer": "-di past is for events you directly witnessed/experienced. -miş past is for reported/hearsay events you didn't see yourself.", "category": "past_tense", "level": "B1", "difficulty": "hard"},
    {"question": "How do you say 'I cannot go' in Turkish?", "expected_answer": "Gidemiyorum (git + eme + iyor + um, negation of -ebil ability)", "category": "negation", "level": "B1", "difficulty": "hard"},
    {"question": "What does 'Afiyet olsun' mean?", "expected_answer": "It means 'Enjoy your meal' or 'Bon appétit'", "category": "phrases", "level": "A1", "difficulty": "easy"},
    {"question": "What are the front vowels in Turkish?", "expected_answer": "e, i, ö, ü are the front vowels", "category": "vowel_harmony", "level": "A1", "difficulty": "easy"},
    {"question": "How do you say 'from Istanbul' in Turkish?", "expected_answer": "İstanbul'dan (İstanbul + -dan, ablative case, 'a' is back vowel → -dan)", "category": "case_suffixes", "level": "A2", "difficulty": "medium"},
    {"question": "What does the suffix -siz/-sız mean?", "expected_answer": "It means 'without': şekersiz = sugarless, işsiz = unemployed/without work", "category": "suffixes", "level": "A2", "difficulty": "medium"},
    {"question": "Translate: 'Ne yapıyorsun?'", "expected_answer": "What are you doing?", "category": "translation", "level": "A1", "difficulty": "easy"},
    {"question": "How do you say 'I must go' in Turkish?", "expected_answer": "Gitmeliyim (git + meli + yim, obligation suffix -meli)", "category": "modality", "level": "B1", "difficulty": "hard"},
    {"question": "What is special about the letter 'ğ' in Turkish?", "expected_answer": "The 'soft g' (ğ) is nearly silent — it lengthens the preceding vowel. Example: dağ (mountain) is pronounced 'daa'", "category": "pronunciation", "level": "A1", "difficulty": "medium"},
    {"question": "How do you say 'Turkey's capital' in Turkish?", "expected_answer": "Türkiye'nin başkenti (Türkiye + genitive -nin + başkent + possessive -i)", "category": "possession", "level": "B1", "difficulty": "hard"},
]


# ═══ HuggingFace Fetcher (optional) ══════════════════════════════════════════

def fetch_from_huggingface() -> List[Dict]:
    """Optionally fetch TQuAD data from HuggingFace datasets."""
    try:
        from datasets import load_dataset
        print("  📡 Fetching TQuAD from HuggingFace...")
        dataset = load_dataset("erdometo/tquad", split="train")
        entries = []
        seen_topics = set()
        for item in dataset:
            title = item.get("title", "")
            context = item.get("context", "")
            if title and context and title not in seen_topics and len(context) > 100:
                seen_topics.add(title)
                entries.append({
                    "topic": f"TQuAD: {title[:60]}",
                    "content": context[:3000],
                    "source": "tquad_huggingface",
                    "language": "turkish",
                    "harvested_at": datetime.now().isoformat(),
                })
                if len(entries) >= 50:
                    break
        print(f"  ✅ Fetched {len(entries)} entries from TQuAD")
        return entries
    except ImportError:
        print("  ⚠️  'datasets' library not installed. Run: pip install datasets")
        return []
    except Exception as e:
        print(f"  ⚠️  HuggingFace fetch failed: {e}")
        return []


# ═══ Build Knowledge Base ══════════════════════════════════════════════════════

def build_knowledge_base(fetch_hf: bool = False) -> List[Dict]:
    """Compile the full Turkish knowledge base from all sources."""
    entries = []

    # 1. Grammar rules
    print("\n  📖 Loading grammar rules...")
    for rule in GRAMMAR_RULES:
        entries.append({
            "topic": rule["topic"],
            "content": rule["content"],
            "level": rule["level"],
            "category": "grammar",
            "source": "curated",
            "harvested_at": datetime.now().isoformat(),
        })
    print(f"  ✅ {len(GRAMMAR_RULES)} grammar topics loaded")

    # 2. Vocabulary lists
    print("  📚 Loading vocabulary lists...")
    vocab_count = 0
    for level, categories in VOCABULARY.items():
        for category, word_pairs in categories.items():
            content_lines = [f"{tr} = {en}" for tr, en in word_pairs]
            entries.append({
                "topic": f"Vocabulary ({level}): {category}",
                "content": "\n".join(content_lines),
                "level": level,
                "category": "vocabulary",
                "source": "curated",
                "harvested_at": datetime.now().isoformat(),
            })
            vocab_count += len(word_pairs)
    print(f"  ✅ {vocab_count} vocabulary items across {len(entries) - len(GRAMMAR_RULES)} categories")

    # 3. HuggingFace (optional)
    if fetch_hf:
        hf_entries = fetch_from_huggingface()
        entries.extend(hf_entries)

    return entries


def save_knowledge(entries: List[Dict]):
    """Save knowledge base to JSON."""
    os.makedirs(DATA_DIR, exist_ok=True)
    data = {
        "language": "Turkish",
        "description": "Turkish language learning knowledge base for the AI tutor agent",
        "total_topics": len(entries),
        "generated_at": datetime.now().isoformat(),
        "knowledge_base": entries,
    }
    with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Saved {len(entries)} topics to {KNOWLEDGE_FILE}")


def save_test_qa():
    """Save the test Q&A dataset."""
    os.makedirs(DATA_DIR, exist_ok=True)
    data = {
        "language": "Turkish",
        "total_questions": len(TEST_QA),
        "categories": list(set(q["category"] for q in TEST_QA)),
        "levels": list(set(q["level"] for q in TEST_QA)),
        "generated_at": datetime.now().isoformat(),
        "test_pairs": TEST_QA,
    }
    with open(TEST_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  💾 Saved {len(TEST_QA)} test Q&A pairs to {TEST_FILE}")


def show_status():
    """Show knowledge base statistics."""
    print(f"\n{'═' * 55}")
    print(f"  🇹🇷 Turkish Tutor — Dataset Status")
    print(f"{'═' * 55}")
    if os.path.exists(KNOWLEDGE_FILE):
        with open(KNOWLEDGE_FILE, encoding="utf-8") as f:
            kb = json.load(f)
        entries = kb.get("knowledge_base", [])
        levels = {}
        categories = {}
        for e in entries:
            lvl = e.get("level", "?")
            cat = e.get("category", "?")
            levels[lvl] = levels.get(lvl, 0) + 1
            categories[cat] = categories.get(cat, 0) + 1
        print(f"\n  Topics:      {len(entries)}")
        print(f"  By level:    {levels}")
        print(f"  By category: {categories}")
    else:
        print("\n  No knowledge base yet. Run: python dataset.py")
    if os.path.exists(TEST_FILE):
        with open(TEST_FILE, encoding="utf-8") as f:
            qa = json.load(f)
        print(f"  Test Q&A:    {qa.get('total_questions', 0)} questions")
    print(f"\n{'═' * 55}\n")


def main():
    parser = argparse.ArgumentParser(description="🇹🇷 Turkish Tutor — Dataset Pipeline")
    parser.add_argument("--fetch-hf", action="store_true", help="Also fetch from HuggingFace (requires 'datasets' library)")
    parser.add_argument("--status", action="store_true", help="Show dataset statistics")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    print("\n🇹🇷 Building Turkish Language Knowledge Base...\n")
    entries = build_knowledge_base(fetch_hf=args.fetch_hf)
    save_knowledge(entries)
    save_test_qa()
    print(f"\n✅ Dataset ready! Run: python tutor.py\n")


if __name__ == "__main__":
    main()
