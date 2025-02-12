from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Charger le modèle DialoGPT-small et le tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

@app.route('/api', methods=['POST'])
def generate_response():
    data = request.get_json()
    prompt = data.get('message', 'Bonjour, comment puis-je vous aider aujourd\'hui ?')

    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
    output_sequences = model.generate(inputs['input_ids'], max_new_tokens=50, num_return_sequences=1, attention_mask=inputs['attention_mask'])
    response = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    return jsonify({'response': response})

# Fonction handler pour Vercel
def handler(event, context):
    return app(event, context)
