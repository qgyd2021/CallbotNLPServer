from toolbox.pyidaungsu.tokenize import Tokenize


def tokenize(text, lang='mm', form='syllable'):
    return Tokenize().tokenize(text, lang, form)


def demo1():
    text = 'မြန်မာဘာသာ'
    result = tokenize(text)
    print(result)
    return


if __name__ == '__main__':
    demo1()
