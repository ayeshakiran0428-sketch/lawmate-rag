import pytest
from utils import clean_query

def test_clean_query_basic():
    assert clean_query("HELLO") == "hello"

def test_clean_query_slang_replacement():
    assert clean_query("plz help") == "please help"
    assert clean_query("yrr kyu") == "why" # 'yrr' -> '', 'kyu' -> 'why'
    assert clean_query("mera masla") == "my problem"

def test_clean_query_word_boundaries():
    # The old logic would replace 'net' inside 'planet' -> 'plainternet'
    assert clean_query("planet") == "planet"
    assert clean_query("net") == "internet"
    assert clean_query("internet") == "internet"

def test_clean_query_punctuation():
    # We want to keep some punctuation for better translation, e.g., ?
    assert clean_query("What is this?") == "what is this?"
    assert clean_query("Wow!") == "wow!"
    assert clean_query("Hello, world.") == "hello world." # Maybe strip commas/periods or keep them

def test_clean_query_multiple_slangs():
    assert clean_query("yrr mujhe wifi ka masla hai") == "i internet ka problem is"
