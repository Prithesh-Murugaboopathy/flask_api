import fitz  # PyMuPDF
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS to handle Cross-Origin Requests
import torch

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Define a ChunkedDataset class for chunking the input text
class ChunkedDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, file_path='ncert_text.txt', block_size=128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Load the text from the file
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        # Tokenize the entire text
        self.encodings = tokenizer(self.text, truncation=True, padding="max_length", max_length=self.block_size, return_tensors='pt')

    def __getitem__(self, idx):
        # Retrieve the input_ids and attention_mask from the tokenized text
        input_ids = self.encodings['input_ids'][idx]
        attention_mask = self.encodings['attention_mask'][idx]
        
        # For language modeling, labels are the same as input_ids
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids  # For language modeling, the labels are the same as input_ids
        }
    
    def __len__(self):
        return len(self.encodings['input_ids'])

# Fine-tune GPT-2 model
def fine_tune_gpt2(text, model_name='gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Set the pad_token to eos_token (for padding)
    tokenizer.pad_token = tokenizer.eos_token

    # Create ChunkedDataset for fine-tuning
    dataset = ChunkedDataset(tokenizer, file_path='ncert_text.txt')

    # Training Arguments
    training_args = TrainingArguments(
        output_dir='./results',          # Output directory
        num_train_epochs=1,              # Number of training epochs
        per_device_train_batch_size=4,   # Batch size for training
        save_steps=10_000,               # Save checkpoint every 10,000 steps
        save_total_limit=2,              # Limit the number of saved checkpoints
    )

    # Trainer
    trainer = Trainer(
        model=model,                     # The model to train
        args=training_args,              # Training arguments
        train_dataset=dataset,           # Training dataset
    )

    # Fine-tuning
    trainer.train()

    return model, tokenizer

# Define a function to generate a response
def generate_response(model, tokenizer, question):
    inputs = tokenizer.encode_plus(question, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    outputs = model.generate(
        inputs['input_ids'], 
        attention_mask=inputs['attention_mask'], 
        max_length=500, 
        num_return_sequences=1, 
        do_sample=False,  # Ensure deterministic responses (no randomness)
        no_repeat_ngram_size=2,  # Prevent repeating phrases in the generated text
        top_k=50,  # Top-k sampling (limit to the top 50 tokens)
        top_p=1.0,  # Nucleus sampling (consider all tokens)
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Route to fine-tune the model with text extracted from a PDF
@app.route('/fine-tune', methods=['POST'])
def fine_tune():
    data = request.json  # Get the JSON payload
    pdf_path = data.get('pdf_path')  # Get the PDF file path from the request

    if pdf_path:
        text = extract_text_from_pdf(pdf_path)
        
        # Save the extracted text to a file
        with open('ncert_text.txt', 'w', encoding='utf-8') as f:
            f.write(text)

        # Fine-tune the model
        model, tokenizer = fine_tune_gpt2(text, model_name='gpt2')

        return jsonify({'message': 'Model fine-tuned successfully!'}), 200
    else:
        return jsonify({'error': 'No PDF file path provided'}), 400

# Route to generate a response from the model
@app.route('/generate', methods=['POST'])
def generate():
    data = request.json  # Get the JSON payload
    question = data.get('question')  # Get the question from the request

    if question:
        response = generate_response(model, tokenizer, question)
        return jsonify({'response': response}), 200
    else:
        return jsonify({'error': 'No question provided'}), 400

# Main script
if __name__ == '__main__':
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Set the pad_token to eos_token
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Start Flask app
    app.run(debug=True)
