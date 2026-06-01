import re


def clean_query(text):
    text = text.lower()

    slang_map = {
        "plz": "please",
        "pls": "please",
        "yrr": "",
        "yar": "",
        "ni": "not",
        "nhi": "not",
        "nai": "not",
        "kr": "do",
        "kro": "do",
        "kar": "do",
        "masla": "problem",
        "issue": "problem",
        "net": "internet",
        "wifi": "internet",
        "kya": "what",
        "kyu": "why",
        "kaise": "how",
        "mujhe": "i",
        "mera": "my",
        "hai": "is",
        "hain": "are"
    }

    # Replace using word boundaries to avoid partial word replacements
    for k, v in slang_map.items():
        # \b matches word boundary
        pattern = r'\b' + re.escape(k) + r'\b'
        # if v is empty we should also try to strip extra spaces left behind
        text = re.sub(pattern, v, text)
    
    # Clean up multiple spaces that might have been created by replacing with empty string
    text = re.sub(r'\s+', ' ', text).strip()

    # Old regex was: text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # New regex: allow basic punctuation like ?, !, . for better translation context
    text = re.sub(r'[^a-zA-Z0-9\s\?!\.]', '', text)
    
    return text

def translate_to_english(text):
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return translated if translated else text
    except Exception as e:
        print(f"Translation error: {e}")
        return text
