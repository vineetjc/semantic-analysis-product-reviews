import unicodedata
from unidecode import unidecode

def deEmojify(inputString):
    returnString = ""

    for character in inputString:
        try:
            character.encode("ascii")
            returnString += character
        except UnicodeEncodeError:
            replaced = unidecode(str(character))
            if replaced != '':
                returnString += replaced
            else:
                try:
                     returnString += " " + unicodedata.name(character) + " "
                except ValueError:
                     returnString += "[x]"

    return returnString


line='😊just checking❤️😊'
print (deEmojify(line))