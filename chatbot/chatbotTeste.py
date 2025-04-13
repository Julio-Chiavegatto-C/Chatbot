from chatterbot import ChatBot

from chatterbot.trainers import ListTrainer

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

while True:
    try:
        bot_input = chatbot.get_response(input())
        print(bot_input)

    except(KeyboardInterrupt, EOFError, SystemExit):
        break