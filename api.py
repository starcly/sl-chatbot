import logging
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration du logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Charger le modèle DialoGPT-small et le tokenizer
logger.error("Chargement du modèle et du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    logger.error("Token de padding défini.")

@app.route('/api', methods=['POST'])
def generate_response():
    logger.error("Requête reçue sur /api")
    try:
        data = request.get_json()
        if not data:
            logger.error("Aucune donnée JSON reçue")
            return jsonify({'error': 'No JSON data received'}), 400

        prompt = data.get('message', 'Bonjour, comment puis-je vous aider aujourd\'hui ?')
        logger.error(f"Prompt reçu: {prompt[:50]}")  # Limiter la longueur du log

        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=256, padding='max_length')
        output_sequences = model.generate(inputs['input_ids'], max_new_tokens=30, num_return_sequences=1, attention_mask=inputs['attention_mask'])
        response = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        logger.error(f"Réponse générée: {response[:50]}")  # Limiter la longueur du log

        return jsonify({'response': response})

    except Exception as e:
        logger.error(f"Erreur lors du traitement de la requête: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

# Ajout d'une route de test
@app.route('/test', methods=['GET'])
def test():
    logger.error("Accès à la route de test /test")
    return jsonify(message="Test route is working!")

# Fonction handler pour Vercel
def handler(event, context):
    logger.error("Nouvelle requête reçue par le handler")
    return app(event, context)

if __name__ == '__main__':
    logger.error("Démarrage de l'application Flask...")
    app.run(host='0.0.0.0', port=5000)
