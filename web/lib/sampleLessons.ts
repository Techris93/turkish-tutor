import { SavedLesson } from "./learning";

export const SAMPLE_LESSONS: SavedLesson[] = [
  {
    id: "sample-b1-nouns",
    title: "B1 Core Nouns (İsimler)",
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    result: {
      source_type: "image",
      source_label: "b1_nouns_list.jpg",
      inferred_level: "B1",
      study_level: "B1",
      target_language: "English",
      preview: "akıcı akran aksan alaylı alt yazı araştırma arzu avantaj ayaküstü yemek bağlantı baskı başlangıç beceri alanları",
      units: [
        { text: "akıcı", kind: "word", turkish_signal: true },
        { text: "akran", kind: "word", turkish_signal: true },
        { text: "aksan", kind: "word", turkish_signal: true }
      ],
      vocabulary_warning: "",
      note: "### B1 Nouns Study Guide\nThis lesson covers common B1 Turkish nouns that frequently appear in reading and conversations.\n\n* **Nouns & Word Forms:** Focus on compound structures (e.g. *alt yazı*, *beceri alanları*).\n* **Usage Tips:** *Ayaküstü yemek* literally translates to 'standing food' and is used to describe fast food or eating on the run.",
      vocabulary_cards: [
        {
          turkish: "akıcı",
          item_type: "noun",
          translation: "fluent / flowing",
          cefr_level: "B1",
          example_tr: "O, Türkçe'yi çok akıcı bir şekilde konuşuyor.",
          example_translation: "He speaks Turkish in a very fluent manner.",
          learner_note: "Can be used as a noun or adjective.",
          tts_word: "akıcı",
          tts_sentence: "O, Türkçe'yi çok akıcı bir şekilde konuşuyor."
        },
        {
          turkish: "akran",
          item_type: "noun",
          translation: "peer / contemporary",
          cefr_level: "B1",
          example_tr: "Çocuklar akranları ile oynamayı severler.",
          example_translation: "Children love to play with their peers.",
          learner_note: "Refers to people of the same age.",
          tts_word: "akran",
          tts_sentence: "Çocuklar akranları ile oynamayı severler."
        },
        {
          turkish: "aksan",
          item_type: "noun",
          translation: "accent",
          cefr_level: "B1",
          example_tr: "Onun hafif bir İngiliz aksanı var.",
          example_translation: "He has a slight English accent.",
          learner_note: "Commonly used for foreign language pronunciation.",
          tts_word: "aksan",
          tts_sentence: "Onun hafif bir İngiliz aksanı var."
        },
        {
          turkish: "alaylı",
          item_type: "noun",
          translation: "self-taught professional",
          cefr_level: "B1",
          example_tr: "O, okul okumadı ama alaylı bir mühendis.",
          example_translation: "He didn't go to school but he is a self-taught engineer.",
          learner_note: "Refers to learning a trade by experience rather than formal education.",
          tts_word: "alaylı",
          tts_sentence: "O, okul okumadı ama alaylı bir mühendis."
        },
        {
          turkish: "alt yazı",
          item_type: "noun",
          translation: "subtitle",
          cefr_level: "B1",
          example_tr: "Yabancı filmleri genellikle alt yazı ile izlerim.",
          example_translation: "I usually watch foreign movies with subtitles.",
          learner_note: "Two words: alt (under) + yazı (writing).",
          tts_word: "alt yazı",
          tts_sentence: "Yabancı filmleri genellikle alt yazı ile izlerim."
        },
        {
          turkish: "araştırma",
          item_type: "noun",
          translation: "research / investigation",
          cefr_level: "B1",
          example_tr: "Yeni araştırma sonuçları yarın açıklanacak.",
          example_translation: "The new research results will be announced tomorrow.",
          learner_note: "Verb form is 'araştırmak'.",
          tts_word: "araştırma",
          tts_sentence: "Yeni araştırma sonuçları yarın açıklanacak."
        },
        {
          turkish: "arzu",
          item_type: "noun",
          translation: "desire / wish",
          cefr_level: "B1",
          example_tr: "Onun tek arzusu başarılı olmaktır.",
          example_translation: "His only desire is to be successful.",
          learner_note: "Synonym with 'istek'.",
          tts_word: "arzu",
          tts_sentence: "Onun tek arzusu başarılı olmaktır."
        },
        {
          turkish: "avantaj",
          item_type: "noun",
          translation: "advantage",
          cefr_level: "B1",
          example_tr: "Dil bilmek iş hayatında büyük bir avantajdır.",
          example_translation: "Knowing a language is a big advantage in business life.",
          learner_note: "Borrowed from French 'avantage'.",
          tts_word: "avantaj",
          tts_sentence: "Dil bilmek iş hayatında büyük bir avantajdır."
        },
        {
          turkish: "ayaküstü yemek",
          item_type: "noun",
          translation: "fast food / eating on the run",
          cefr_level: "B1",
          example_tr: "Sağlığımız için ayaküstü yemekten kaçınmalıyız.",
          example_translation: "We should avoid fast food for our health.",
          learner_note: "Refers to eating quickly while standing or on the go.",
          tts_word: "ayaküstü yemek",
          tts_sentence: "Sağlığımız için ayaküstü yemekten kaçınmalıyız."
        },
        {
          turkish: "bağlantı",
          item_type: "noun",
          translation: "connection / link",
          cefr_level: "B1",
          example_tr: "İnternet bağlantısı çok yavaş görünüyor.",
          example_translation: "The internet connection seems very slow.",
          learner_note: "Verb root is 'bağlamak' (to tie/link).",
          tts_word: "bağlantı",
          tts_sentence: "İnternet bağlantısı çok yavaş görünüyor."
        },
        {
          turkish: "baskı",
          item_type: "noun",
          translation: "pressure / printing",
          cefr_level: "B1",
          example_tr: "Kitabın ilk baskısı tükendi.",
          example_translation: "The first print edition of the book is sold out.",
          learner_note: "Can mean physical pressure, mental stress, or printing.",
          tts_word: "baskı",
          tts_sentence: "Kitabın ilk baskısı tükendi."
        },
        {
          turkish: "başlangıç",
          item_type: "noun",
          translation: "beginning / start",
          cefr_level: "B1",
          example_tr: "Bu yeni bir başlangıç olacak.",
          example_translation: "This will be a new start.",
          learner_note: "Noun form of 'başlamak' (to start).",
          tts_word: "başlangıç",
          tts_sentence: "Bu yeni bir başlangıç olacak."
        },
        {
          turkish: "beceri alanları",
          item_type: "noun",
          translation: "skill areas / fields of competence",
          cefr_level: "B1",
          example_tr: "Özgeçmişinizde beceri alanlarınızı belirtin.",
          example_translation: "Specify your skill areas on your resume.",
          learner_note: "Compound noun: beceri (skill) + alanları (fields of).",
          tts_word: "beceri alanları",
          tts_sentence: "Özgeçmişinizde beceri alanlarınızı belirtin."
        }
      ]
    }
  },
  {
    id: "sample-a1-convo",
    title: "A1 Daily Conversations (Günlük Konuşma)",
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    result: {
      source_type: "text",
      source_label: "A1 Conversation Starter",
      inferred_level: "A1",
      study_level: "A1",
      target_language: "English",
      preview: "merhaba nasılsın teşekkürler hoş geldiniz görüşürüz memnun oldum",
      units: [
        { text: "merhaba", kind: "word", turkish_signal: true },
        { text: "nasılsın", kind: "word", turkish_signal: true }
      ],
      vocabulary_warning: "",
      note: "### A1 Greetings and Daily Conversation Chunks\nLearn essential expressions for daily contact in Turkish.\n\n* **Responses:** When someone says *Hoş geldiniz* (Welcome), respond with *Hoş bulduk* (We found it pleasant / We feel welcome).\n* **Politeness:** *Teşekkür ederim* is the formal way to say thank you.",
      vocabulary_cards: [
        {
          turkish: "merhaba",
          item_type: "phrase",
          translation: "hello",
          cefr_level: "A1",
          example_tr: "Merhaba, benim adım Ahmet.",
          example_translation: "Hello, my name is Ahmet.",
          learner_note: "Universal greeting, polite and friendly.",
          tts_word: "merhaba",
          tts_sentence: "Merhaba, benim adım Ahmet."
        },
        {
          turkish: "nasılsın",
          item_type: "phrase",
          translation: "how are you?",
          cefr_level: "A1",
          example_tr: "Selam Can, bugün nasılsın?",
          example_translation: "Hi Can, how are you today?",
          learner_note: "Informal. For formal/plural use 'nasılsınız'.",
          tts_word: "nasılsın",
          tts_sentence: "Selam Can, bugün nasılsın?"
        },
        {
          turkish: "teşekkür ederim",
          item_type: "phrase",
          translation: "thank you",
          cefr_level: "A1",
          example_tr: "Yardımın için teşekkür ederim.",
          example_translation: "Thank you for your help.",
          learner_note: "Formal. Informal is 'teşekkürler'.",
          tts_word: "teşekkür ederim",
          tts_sentence: "Yardımın için teşekkür ederim."
        },
        {
          turkish: "hoş geldiniz",
          item_type: "phrase",
          translation: "welcome",
          cefr_level: "A1",
          example_tr: "Evimize hoş geldiniz arkadaşlar.",
          example_translation: "Welcome to our house friends.",
          learner_note: "Standard welcoming phrase. Reply with 'Hoş bulduk'.",
          tts_word: "hoş geldiniz",
          tts_sentence: "Evimize hoş geldiniz arkadaşlar."
        },
        {
          turkish: "görüşürüz",
          item_type: "phrase",
          translation: "see you / goodbye",
          cefr_level: "A1",
          example_tr: "Yarin okulda görüşürüz.",
          example_translation: "See you at school tomorrow.",
          learner_note: "Literally 'we will see each other'. Used as a casual farewell.",
          tts_word: "görüşürüz",
          tts_sentence: "Yarin okulda görüşürüz."
        },
        {
          turkish: "memnun oldum",
          item_type: "phrase",
          translation: "nice to meet you",
          cefr_level: "A1",
          example_tr: "Tanıştığımıza memnun oldum.",
          example_translation: "I am pleased to meet you.",
          learner_note: "Reply with 'Ben de memnun oldum' (I am pleased too).",
          tts_word: "memnun oldum",
          tts_sentence: "Tanıştığımıza memnun oldum."
        }
      ]
    }
  },
  {
    id: "sample-a2-restaurant",
    title: "A2 Dining & Ordering (Restoranda)",
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    result: {
      source_type: "image",
      source_label: "restaurant_menu.jpg",
      inferred_level: "A2",
      study_level: "A2",
      target_language: "English",
      preview: "garson hesap lütfen menü sipariş vermek lezzetli afiyet olsun",
      units: [
        { text: "garson", kind: "word", turkish_signal: true },
        { text: "menü", kind: "word", turkish_signal: true }
      ],
      vocabulary_warning: "",
      note: "### A2 Ordering Food and Dining Out\nEssential vocabulary for ordering meals and communicating in restaurants.\n\n* **Asking for the Bill:** Say *Hesap lütfen* (Bill please) or *Hesabı alabilir miyim?* (Can I have the bill?).\n* **Eating Etiquette:** When someone starts eating, say *Afiyet olsun* (Bon appétit / enjoy your meal).",
      vocabulary_cards: [
        {
          turkish: "garson",
          item_type: "noun",
          translation: "waiter / waitress",
          cefr_level: "A2",
          example_tr: "Garson bey, bakar mısınız lütfen?",
          example_translation: "Excuse me waiter, could you look here please?",
          learner_note: "Polite address is 'Garson bey' or simply 'Pardon'.",
          tts_word: "garson",
          tts_sentence: "Garson bey, bakar mısınız lütfen?"
        },
        {
          turkish: "hesap lütfen",
          item_type: "phrase",
          translation: "the bill, please",
          cefr_level: "A2",
          example_tr: "Yemek çok güzeldi, hesap lütfen.",
          example_translation: "The food was very good, bill please.",
          learner_note: "Polite standard request for payment.",
          tts_word: "hesap lütfen",
          tts_sentence: "Yemek çok güzeldi, hesap lütfen."
        },
        {
          turkish: "menü",
          item_type: "noun",
          translation: "menu",
          cefr_level: "A2",
          example_tr: "Yemek seçmek için menüyü inceledik.",
          example_translation: "We examined the menu to choose food.",
          learner_note: "Borrowed from French 'menu'.",
          tts_word: "menü",
          tts_sentence: "Yemek seçmek için menüyü inceledik."
        },
        {
          turkish: "sipariş vermek",
          item_type: "verb",
          translation: "to place an order",
          cefr_level: "A2",
          example_tr: "Köfte ve salata sipariş vermek istiyorum.",
          example_translation: "I would like to order meatballs and salad.",
          learner_note: "Compound verb with 'vermek' (to give).",
          tts_word: "sipariş vermek",
          tts_sentence: "Köfte ve salata sipariş vermek istiyorum."
        },
        {
          turkish: "lezzetli",
          item_type: "adjective",
          translation: "delicious / tasty",
          cefr_level: "A2",
          example_tr: "Bu çorba gerçekten çok lezzetli.",
          example_translation: "This soup is really very delicious.",
          learner_note: "Derived from 'lezzet' (taste/flavor) + suffix '-li'.",
          tts_word: "lezzetli",
          tts_sentence: "Bu çorba gerçekten çok lezzetli."
        },
        {
          turkish: "afiyet olsun",
          item_type: "phrase",
          translation: "bon appétit / enjoy your meal",
          cefr_level: "A2",
          example_tr: "Yemeğiniz geldi, afiyet olsun.",
          example_translation: "Your food has arrived, enjoy your meal.",
          learner_note: "Said before, during, or after eating to wish good digestion.",
          tts_word: "afiyet olsun",
          tts_sentence: "Yemeğiniz geldi, afiyet olsun."
        }
      ]
    }
  }
];
