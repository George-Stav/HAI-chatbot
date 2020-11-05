import os
from question import answer

prompt = lambda x: input(f'{x}\n>')

USERNAME = prompt('What should I call you?')
BOTNAME = prompt('What is my name?')

userInput = ''
print('\nEnter \'q\' to quit any time.\n')

while userInput != 'q':
    userInput = prompt('Listening...')
    print(answer(userInput))