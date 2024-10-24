import os
import fitz  # PyMuPDF
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from flask import Flask, request, jsonify
from flask_cors import CORS





# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

@app.route('/')
def home():
    return 'Hello, Flask!'

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Fine-tune GPT-2 model
def fine_tune_gpt2(text, model_name='gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Save the text to a file with utf-8 encoding
    with open('ncert_text.txt', 'w', encoding='utf-8') as f:
        f.write(text)

    # Create a dataset
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path='ncert_text.txt',
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    return model, tokenizer

# Load your PDF and fine-tune the model
pdf_path = 'ncert_textbook.pdf'  # Path to your PDF
text = extract_text_from_pdf(pdf_path)
model, tokenizer = fine_tune_gpt2(text)

# Function to generate a response
def generate_response(model, tokenizer, question):
    tokenizer.pad_token = tokenizer.eos_token  # Set the pad token to eos token
    
    inputs = tokenizer.encode_plus(question, return_tensors='pt', padding=True)
    
    outputs = model.generate(
        inputs['input_ids'], 
        attention_mask=inputs['attention_mask'], 
        max_length=500, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2, 
        early_stopping=False, 
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Define the API endpoint
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question', '')
    response = generate_response(model, tokenizer, question)
    return jsonify({'response': response})

if __name__ == '__main__':
    # Bind to the port specified by the environment variable
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
