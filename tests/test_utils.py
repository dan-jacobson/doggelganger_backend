from doggelganger.utils import valid_link


def test_valid_link():
    assert valid_link("https://www.google.com")
    assert not valid_link("https://thisisnotarealwebsite12345.com")
