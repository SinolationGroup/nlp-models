from random import choice


def uncapitalize(s):
    return s[:1].lower() + s[1:] if s else ""


def keyword_aug(s):
    words = s.split()
    if choice([0, 1]) and len(words) > 5:
        range_top = choice(list(range(6, 15)))
        s = " ".join(words[:range_top])
    if choice([0, 1]):
        s = s.lower()
    else:
        s = uncapitalize(s) if not s[1].isupper() else s
    return s
