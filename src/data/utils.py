from random import choice


def uncapitalize(s: str) -> str:
    """This function uncapitalaze the first character in a text s

    Parameters
    ----------
    s : str
        any text

    Returns
    -------
    str
    """
    return s[:1].lower() + s[1:] if s else ""


def keyword_aug(
    s: str, aug_length: int = 5, min_lenght: int = 6, max_length: int = 15
) -> str:
    """If number of words in text `s` more than `aug_length` then text will be cut randomly from `min_lenght` to `max_length` words

    Parameters
    ----------
    s : str
        any text
    aug_length : int, optional
        by default 5
    min_lenght : int, optional
        by default 6
    max_length : int, optional
        by default 15

    Returns
    -------
    str
        augmented text
    """

    words = s.split()
    if choice([0, 1]) and len(words) > aug_length:
        range_top = choice(list(range(min_lenght, max_length)))
        s = " ".join(words[:range_top])
    if choice([0, 1]):
        s = s.lower()
    else:
        s = uncapitalize(s) if not s[1].isupper() else s
    return s
