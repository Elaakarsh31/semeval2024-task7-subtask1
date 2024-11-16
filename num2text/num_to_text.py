import pandas as pd
import re

from num2words import num2words

def convert_to_text(question):
    # pre process the string
    question = re.sub(r'[^a-zA-Z0-9\s.]', '', question)
    
    # convert decimals 
    for word in question.split(" "):
        # get all decimals
        decimals = re.findall(r'\b\d+\.\d+\b', word)
        if word in decimals:
            question = question.replace(word, num2words(float(word)))
    
    # get all other numbers
    numbers = re.findall(r'([^\d]|^)(\d+)([^\d]|$)', question)
    for number in numbers:
        num = number[1]
        text = num2words(int(num))
        question = re.sub(rf'([^\d]|^){num}([^\d]|$)', rf'\1{text}\2', question)
    
    return question


if __name__ == '__main__':
    # TODO: replace with your file path
    file_path = 'QQA_train.json'
    # TODO: replace with your column name
    column_name = 'question'

    # read json
    data = pd.read_json(path_or_buf=file_path)
    data['question_text'] = data[column_name].apply(func=convert_to_text)
    # save new file in CSV
    data.to_json(f'{file_path.split(".")[0]}.csv', index=False)
