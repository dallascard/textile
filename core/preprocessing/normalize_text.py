import re
import ftfy
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


def fix_web_text(text):
    """Replace certain strings that appear in web text with more sesible alternatives
    >>> fix_web_text("Senate&rsquo;s")
    "Senate's"
    """
    #text = re.sub('&rsquo;', "'", text)
    text = ftfy.fix_text(text)
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


def split_off_punctuation(text):
    """ Put spaces between punctuation and characters
    
    >>> split_off_punctuation("I'm a sentence.")
    "I ' m a sentence ."
    >>> split_off_punctuation('This is a "sentence".')
    'This is a " sentence " .'
    >>> split_off_punctuation("It's 12 minutes 'til 4:00!")
    "It ' s 12 minutes ' til 4 : 00 !"
    """
    text = re.findall(r"\w+|[^\w\s]", text)
    return ' '.join(text)


doctest.testmod()
