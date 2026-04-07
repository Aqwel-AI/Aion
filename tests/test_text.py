import aion


def test_count_words():
    assert aion.text.count_words("hello world  from aion") == 4


def test_extract_emails():
    found = aion.text.extract_emails("Contact: test@example.com or admin@aion.ai")
    assert found == ["test@example.com", "admin@aion.ai"]


def test_is_palindrome():
    assert aion.text.is_palindrome("A man, a plan, a canal: Panama")


def test_detect_language():
    assert aion.text.detect_language("the quick brown fox and the dog") == "en"


def test_generate_hash_sha256():
    digest = aion.text.generate_hash("aion", "sha256")
    assert len(digest) == 64
