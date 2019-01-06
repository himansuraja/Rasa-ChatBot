from rasa_core.agent import Agent
from rasa_core.interpreter import RasaNLUInterpreter

interpreter = RasaNLUInterpreter('models/current/nlu')
agent = Agent.load('models/dialogue', interpreter=interpreter)

print("Your bot is ready to talk! Type your messages here or send 'stop'")

while True:
    a = input()
    if a == 'stop':
        break
    responses = agent.handle_message(a)
    for response in responses:
        print(response["text"])
        