from chatterbot import ChatBot

from chatterbot.trainers import ListTrainer

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

chatbot = ChatBot(
    'FamilIA',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        'chatterbot.logic.MathematicalEvaluation',
        'chatterbot.logic.TimeLogicAdapter'
    ],
    database_uri='sqlite:///database.sqlite3'
)

conversation = [
    "Olá",
    "Bom dia! Como posso ajudar?",
    "Quem é você?",
    "Eu sou a FamilIA!",
    "O que você faz?",
    "Eu te ajudo a tirar suas dúvidas!",
    "Entendo, muito obrigado",
    "Eu que agradeço!"
]

trainer = ListTrainer(chatbot)
trainer.train(conversation)

# modelo da Hugging Face
model_name = "distilbert-base-uncased"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


def detectar_intencao(frase):
    inputs = tokenizer(frase, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    return predicted_class

# interacoes
intencoes = {
    0: "Agendar consulta",
    1: "Cancelar consulta",
    2: "Remarcar consulta",
    3: "Dúvida geral"
}

print("FamilIA iniciada. Digite sua mensagem:")

while True:
    try:
        user_input = input()

        
        intencao_detectada = detectar_intencao(user_input)
        descricao_intencao = intencoes.get(intencao_detectada, "Intenção desconhecida")

        print(f"[Intenção Detectada]: {descricao_intencao}")

        if descricao_intencao == "Agendar consulta":
            print("Perfeito! Para agendar sua consulta, me informe a data e o horário desejados.")
        elif descricao_intencao == "Cancelar consulta":
            print("Tudo bem, me diga qual consulta você gostaria de cancelar.")
        elif descricao_intencao == "Remarcar consulta":
            print("Claro! Vamos remarcar. Me informe a nova data e horário.")
        else:
            bot_input = chatbot.get_response(user_input)
            print(bot_input)

    except (KeyboardInterrupt, EOFError, SystemExit):
        print("\nEncerrando o chatbot. Até logo!")
        break
