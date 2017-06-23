import re
import doctest


def strip_html(text):
    """Remove each pair of angle brackets and everything within them

    >>> strip_html('<p>Test</p>')
    'Test'
    >>> strip_html('No HTML')
    'No HTML'
    >>> strip_html('>This is OK<')
    '>This is OK<'
    """
    text = re.sub('<[^>]+>', '', text)
    return text


def replace_underscores(text, replacement='-'):
    """ Replace undescores with a different character

    >>> replace_underscores('filename_01.txt')
    'filename-01.txt'
    >>> replace_underscores('___')
    '---'
    >>> replace_underscores('__init__', replacement='*')
    '**init**'
    """
    return re.sub('_', replacement, text)


doctest.testmod()
