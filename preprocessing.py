import re, string
from pyvi import ViTokenizer, ViPosTagger
import emoji
from file_loader import FileStore, FileReader, DataLoader

def give_emoji_free_text(text):
    allchars = [str for str in text.decode('utf-8')]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.decode('utf-8').split() if not any(i in str for i in emoji_list)])
    return clean_text

def remove_icon(text):
    text = text.lower()
    s = ''
    pattern = r"[a-zA-ZaăâbcdđeêghiklmnoôơpqrstuưvxyàằầbcdđèềghìklmnòồờpqrstùừvxỳáắấbcdđéếghíklmnóốớpqrstúứvxýảẳẩbcdđẻểghỉklmnỏổởpqrstủửvxỷạặậbcdđẹệghịklmnọộợpqrstụựvxỵãẵẫbcdđẽễghĩklmnõỗỡpqrstũữvxỹAĂÂBCDĐEÊGHIKLMNOÔƠPQRSTUƯVXYÀẰẦBCDĐÈỀGHÌKLMNÒỒỜPQRSTÙỪVXỲÁẮẤBCDĐÉẾGHÍKLMNÓỐỚPQRSTÚỨVXÝẠẶẬBCDĐẸỆGHỊKLMNỌỘỢPQRSTỤỰVXỴẢẲẨBCDĐẺỂGHỈKLMNỎỔỞPQRSTỦỬVXỶÃẴẪBCDĐẼỄGHĨKLMNÕỖỠPQRSTŨỮVXỸ,._]"
    
    for char in text:
        if char !=' ':
            if len(re.findall(pattern, char)) != 0:
                s+=char
            elif char == '_':
                s+=char
        else:
            s+=char
    s = re.sub('\\s+',' ',s).strip()
    return s.strip()

def remove_stop_word(text, stopWordList):
    word_lst = text.split(' ')
    filtered_sentence =[]
    for w in word_lst:
        if w not in stopWordList:
            filtered_sentence.append(w)
    return " ".join(filtered_sentence)


def processing(text, stopwordLst):
    text = str(text).strip()
    text = re.sub(r"\s+"," ", text)
    text = remove_icon(text)
    text = ViTokenizer.tokenize(text)
    text = remove_stop_word(text, stopwordLst)
    for punc in string.punctuation:
        if punc != "_":
            text = text.replace(punc, "")
    text = text.strip()
    text = re.sub("\\s+"," ", text).lower()
    return text