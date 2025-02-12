import logging
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration du logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Charger le modèle DialoGPT-small et le tokenizer
print("Chargement du modèle et du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Token de padding défini.")

@app.route('/api', methods=['POST'])
def generate_response():
    print("Requête reçue sur /api")
    try:
        data = request.get_json()
        if not data:
            print("Aucune donnée JSON reçue")
            return jsonify({'error': 'No JSON data received'}), 400

        prompt = data.get('message', 'Bonjour, comment puis-je vous aider aujourd\'hui ?')
        print(f"Prompt reçu: {prompt}")

        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
        output_sequences = model.generate(inputs['input_ids'], max_new_tokens=50, num_return_sequences=1, attention_mask=inputs['attention_mask'])
        response = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        print(f"Réponse générée: {response}")

        return jsonify({'response': response})

    except Exception as e:
        print(f"Erreur lors du traitement de la requête: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

# Ajout d'une route de test
@app.route('/test', methods=['GET'])
def test():
    print("Accès à la route de test /test")
    return jsonify(message="Test route is working!")

# Fonction handler pour Vercel
def handler(event, context):
    print("Nouvelle requête reçue par le handler")
    return app(event, context)

if __name__ == '__main__':
    print("Démarrage de l'application Flask...")
    app.run(host='0.0.0.0', port=5000)
