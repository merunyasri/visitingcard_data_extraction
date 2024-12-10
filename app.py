from flask import Flask, request, jsonify, render_template
import os
import easyocr
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image, ImageDraw
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

model = load_model("NER-VC_model.h5")
# Data loading and preprocessing
data_df = pd.read_csv("ner_dataset - ner_dataset (2).csv", encoding="iso-8859-1", header=0)

# Forward fill missing Sentence Number values
data_df = data_df.ffill()

# Optional: Extract numeric part from 'Sentence: X' or leave as is if already an integer
def extract_sentence_number(val):
    if isinstance(val, str):
        return int(val.split(":")[-1])  # Extract number if it's in string format
    return val  # If it's already an integer, return as is

data_df["Sentence"] = data_df["Sentence"].apply(extract_sentence_number)

# Group words and tags by sentence
sentences = data_df.groupby("Sentence")["Word"].apply(list).tolist()
labels = data_df.groupby("Sentence")["Tag"].apply(list).tolist()

# Vocabulary and tag mapping
tags = ["O", "ORG", "STR", "CIT", "STA", "PER"]  # Include 'O' tag to avoid KeyError
tag2index = {tag: idx for idx, tag in enumerate(tags)}
index2tag = {idx: tag for tag, idx in tag2index.items()}

# Tokenizer for the words
tokenizer = Tokenizer(num_words=None, oov_token="--UNKNOWN_WORD--")
tokenizer.fit_on_texts(sentences)

word2index = tokenizer.word_index
index2word = tokenizer.index_word

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the OCR reader
reader = easyocr.Reader(['en'])

def draw_boxes(image, bounds, color='yellow', width=2):
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    return image

def correct_email(email):
    email = email.replace(" @ ", "@").replace(" .", ".")
    email = email.replace(" @", "@").replace("@ ", "@")
    email = email.replace(" .", ".").replace(". ", ".")
    common_typos = {
        'gmall.com': 'gmail.com',
        'gmall': 'gmail.com',
        'gmail': 'gmail.com',
        'gmail com': 'gmail.com',
        'gnail.com': 'gmail.com',
        'yaho.com': 'yahoo.com',
        'yaho com': 'yahoo.com',
        'hotmial.com': 'hotmail.com',
        'gmaill.com': 'gmail.com',
        'vsnl': 'vsnl.com',
        'tcchsr' : 'tcchsr.com'
    }
    if "@" in email:
        local_part, domain_part = email.split("@", 1)
        for typo, correction in common_typos.items():
            if typo in domain_part:
                domain_part = correction
        return f"{local_part}@{domain_part}"
    return email
def extract_emails_from_text(text_lines):
    emails = set()
    email_pattern = r'\b[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\s*\b'
    for line in text_lines:
        matches = re.findall(email_pattern, line)
        for match in matches:
            corrected_email = correct_email(match)
            emails.add(corrected_email)
    return list(emails)


def is_valid_phone_number(number):
    cleaned_number = re.sub(r'\D', '', number)
    if len(cleaned_number) < 10:
        return False
    valid_patterns = [
        r'^\+?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{4}[-.\s]?\d{4,9}$',
        r'^\d{10}$',
        r'^\d{3}[-.\s]?\d{3}[-.\s]?\d{4}$',
        r'^\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4,7}$'
    ]
    for pattern in valid_patterns:
        if re.fullmatch(pattern, number):
            return True
    return False
def extract_phone_numbers_from_text(text_lines):
    phone_numbers = set()
    phone_patterns = [
        r'\+?\d{1,4}[-.\s]?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{4,9}',  # Matches different formats
    ]
    for line in text_lines:
        for pattern in phone_patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                if is_valid_phone_number(match):
                    phone_numbers.add(match.strip())
    return list(phone_numbers)

role_keywords = [
    'CEO', 'CTO', 'CFO', 'Manager', 'Director', 'Technical Manager',
    'Head', 'Lead', 'President', 'Vice President',
    'VP', 'Chief', 'Designer', 'Paediatrician','Document Writer',
   'Wellness Coach', 'Sales & Service Engineer', 'Dell Store Promoter',
    'Advocate', 'Executive Director', 'Proprietor', 'E.B. Approved & Electrical Contractor',
    'Consulting Civil Engineer', 'Prop', 'Proprietrix', 'Trustee & PRO, Deputy Manager',
    'Country Chief', 'Branch Manager', 'Managing Director', 'Executive Customer Support', 'Training Supervisor',
    'Chairman', 'Founding Partner', 'Business Manager', 'Energy Auditor & Consulting Engineer',
    'Group Head', 'Scientist', 'Asst. Manager', 'Engineer - Sales', 'Project Engineer',
    'Chief Executive', 'Entrepreneur', 'Chief Technology Officer', 'Director - Technical',
    'Technical Director', 'Officer', 'Assistant Manager', 'Research Assosiate', 'Human Resource'
]
role_pattern = r'(?i)\b(?:' + '|'.join(re.escape(keyword) for keyword in role_keywords) + r')\b'

def extract_roles_from_text(text_lines):
    roles = set()
    for line in text_lines:
        matches = re.findall(role_pattern, line)
        roles.update(matches)
    return list(roles)

name_keywords = [
    'SELVAM', 'CHANDRAPPA', 'RAMASAMY' , 'Jayachandru', 'Selvam', 'DEEPA SAMRAJ', 'Sultan Mohideen', 'VinothKumar',
    'Sanjay V.Kale', 'Ravi Kumar', 'Priya Sharma', 'Anil Agarwal', 'Vijay Patil', 'Meenakshi Iyer', 'Nikhil Rao', 'Suresh Babu', 'Radha Krishnan',
    'Pooja Singh', 'Rakesh Mehra', 'Arjun Pillai', 'Harish Menon', 'Nandini Verma', 'Amitabh Joshi', 'Tara Swaminathan', 'Vikram Desai', 
    'Shweta Nair', 'Ajay Prasad', 'Rajesh Kannan', 'Madhavi Rao', 'Girish Kulkarni', 'Sneha Pillai', 'Pradeep Naik',
    'Manoj Tiwari', 'Jyoti Mishra', 'Akash Jain', 'Sunil Kapoor','Ramesh','Murthy','Sathesh','Thangavelu'
]

name_pattern = r'(?i)\b(?:' + '|'.join(re.escape(keyword) for keyword in name_keywords) + r')\b'
def extract_names_from_text(text_lines):
    names = set()
    for line in text_lines:
        matches = re.findall(name_pattern, line)
        names.update(matches)
    return list(names)

company_keywords = [
    'Megatronics', 'Everest Instruments Pvt. Ltd', 'ZAN COMPUTECH', 'Pantech', 'Dell Exclusive Store',
    'Nutrition Centre', 'SVS', 'VEE BEE YARN TEX PRIVATE LIMITED', 'Tata Consultancy Services', 'Infosys Technologies', 
    'Reliance Industries', 'Wipro Ltd.', 'HCL Technologies', 'Tech Mahindra', 'Bajaj Auto', 'Hero MotoCorp', 
    'Asian Paints', 'Bharti Airtel', 'ICICI Bank', 'Axis Bank', 'Adani Group', 'Larsen & Toubro', 'Cipla Ltd.', 
    'Maruti Suzuki', 'Godrej Consumer Products', 'Hindustan Unilever', 'Mahindra & Mahindra', 'BHEL', 'Dr. Reddy\'s Laboratories','Sunshiv','Ashwin Hospital'
]

company_pattern = r'(?i)\b(?:' + '|'.join(re.escape(keyword) for keyword in company_keywords) + r')\b'
def extract_company_names_from_text(text_lines):
    companies = set()
    for line in text_lines:
        matches = re.findall(company_pattern, line)
        companies.update(matches)
    return list(companies)

def extract_pin_codes_from_text(text_lines):
    pin_codes = set()
    pin_code_pattern = r'\b(\d{3})\s*(\d{3})\b'
    for line in text_lines:
        cleaned_line = re.sub(r'\D', '', line)
        if len(cleaned_line) == 6:
            match = re.match(r'(\d{3})(\d{3})', cleaned_line)
            if match:
                pin_code = match.group(1) + match.group(2)
                pin_codes.add(pin_code)
    return list(pin_codes)
def extract_remaining_text(extracted_text, emails_found, phone_numbers_found, roles_found, pin_codes_found):
    for email in emails_found:
        extracted_text = [line.replace(email, '') for line in extracted_text]
    for phone_number in phone_numbers_found:
        extracted_text = [line.replace(phone_number, '') for line in extracted_text]
    for role in roles_found:
        extracted_text = [line.replace(role, '') for line in extracted_text]
    for pincode in pin_codes_found:
        extracted_text=  [line.replace(pincode, '') for line in extracted_text]
    return ' '.join(extracted_text)

def preprocess_remaining_text(text):
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    text = re.sub(r'\+?\d{1,4}[-.\s]?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}', '', text)
    text = re.sub(r'\b(mob|ph|com|email|www|WWW|Website|Phone|Mobile|Email)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'\bwww\S*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b[^\w\s]\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_tags(sentence, model, tokenizer, index2tag):
    words = [word for word in sentence.split() if re.search(r'\w', word)]
    tokenized_sentence = tokenizer.texts_to_sequences([words])
    padded_sentence = pad_sequences(tokenized_sentence, maxlen=model.input_shape[1], padding='post')
    predictions = model.predict(padded_sentence)
    predicted_tags = np.argmax(predictions, axis=-1)[0]
    result = [(word, index2tag[tag_idx]) for word, tag_idx in zip(words, predicted_tags[:len(words)])]
    return result

def organize_ner_results(ner_results):
    persons = []
    organizations = []
    address = []
    for word, tag in ner_results:
        if tag == "PER":
            persons.append(word)
        elif tag == "ORG":
            organizations.append(word)
        elif tag == "CIT":
            address.append(word)
        elif tag == "STR":
            address.append(word)

    return persons, organizations, address

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract_text():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)

    # Read text from image using OCR
    bounds = reader.readtext(image_path)
    extracted_text = [entry[1] for entry in bounds]

    # Extract data
    emails_found = extract_emails_from_text(extracted_text)
    phone_numbers_found = extract_phone_numbers_from_text(extracted_text)
    roles_found = extract_roles_from_text(extracted_text)
    pin_codes_found = extract_pin_codes_from_text(extracted_text)

    remaining_text = extract_remaining_text(extracted_text, emails_found, phone_numbers_found, roles_found, pin_codes_found)
    cleaned_remaining_text = preprocess_remaining_text(remaining_text)
    ner_results = predict_tags(cleaned_remaining_text, model, tokenizer, index2tag)
    persons, organizations, address = organize_ner_results(ner_results)
    # Return extracted information as JSON
    result = {
        'emails': emails_found,
        'phoneNumbers': phone_numbers_found,
        'roles': roles_found,
        'pinCodes': pin_codes_found,
        'name' : persons,
        'company' : organizations,
        'address' : address
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
