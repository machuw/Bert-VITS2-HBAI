import re
import sys

try:
    import MeCab
except ImportError as e:
    raise ImportError("Japanese requires mecab-python3 and unidic-lite.") from e

text = sys.argv[1]

#_TAGGER = MeCab.Tagger("-Owakati")
#parsed = _TAGGER.parse(text).split()[:-1]

_TAGGER = MeCab.Tagger()
parsed = _TAGGER.parse(text)

print(parsed)
