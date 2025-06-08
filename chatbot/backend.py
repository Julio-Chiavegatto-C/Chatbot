from flask import Flask, request, jsonify
from flask_httpauth import HTTPBasicAuth 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from flask_cors import CORS
from flask import send_from_directory


app = Flask(__name__) 
CORS(app) 
auth = HTTPBasicAuth()

chatbot = ChatBot(
    'FamilIA',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        'chatterbot.logic.MathematicalEvaluation',
        'chatterbot.logic.TimeLogicAdapter',
        'chatterbot.logic.BestMatch'
    ],
    database_uri='sqlite:///database.sqlite3'
)

conversation = [
    "Olá", "Bom dia! Como posso ajudar?",
    "Quem é você?", "Eu sou a FamilIA!",
    "O que você faz?", "Eu te ajudo a tirar suas dúvidas!",
    "Entendo, muito obrigado", "Eu que agradeço!"
]

trainer = ListTrainer(chatbot)
trainer.train(conversation)

# Modelo de classificação de intenções
model_name = "distilbert-base-uncased"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Mapeamento de intenções
intencoes = {
    0: "Agendar consulta",
    1: "Cancelar consulta",
    2: "Remarcar consulta",
    3: "Dúvida geral"
}

@auth.verify_password
def verify_password(username, password):
    """Valida usuário e senha para rotas protegidas"""
    return username == "admin" and password == "senha"
@app.route('/')
def serve_frontend():
    pasta = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(pasta, 'test_frontend.html')


@app.route('/api/detect_intent', methods=['POST'])
def detect_intent():
    """Endpoint principal para detecção de intenções"""
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({'error': 'Formato inválido. Envie {"message": "sua_mensagem"}'}), 400
    
    frase = data['message']
    
    # Detecção de intenção
    inputs = tokenizer(frase, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    descricao_intencao = intencoes.get(predicted_class, "Intenção desconhecida")
    
    # Respostas personalizadas
    respostas = {
        "Agendar consulta": "Perfeito! Para agendar sua consulta, me informe a data e o horário desejados.",
        "Cancelar consulta": "Tudo bem, me diga qual consulta você gostaria de cancelar.",
        "Remarcar consulta": "Claro! Vamos remarcar. Me informe a nova data e horário.",
        "Dúvida geral": str(chatbot.get_response(frase))
    }
    
    resposta = respostas.get(descricao_intencao, "Desculpe, não entendi. Poderia reformular?")
    
    return jsonify({
        'intent': descricao_intencao,
        'response': resposta
    })

@app.route('/api/agendamentos', methods=['GET'])
@auth.login_required
def get_agendamentos():
    """Endpoint protegido para listar agendamentos"""
    lista_de_agendamentos = [
        {"id": 1, "paciente": "João Silva", "data": "2025-06-20", "horario": "14:00"},
        {"id": 2, "paciente": "Maria Souza", "data": "2025-06-21", "horario": "10:30"}
    ]
    return jsonify({"agendamentos": lista_de_agendamentos})

@app.route('/')
def home():
    return "Bem-vindo ao Chatbot FamilIA! Use a rota /api/detect_intent para interagir."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 
