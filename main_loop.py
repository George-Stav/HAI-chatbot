import os

prompt = lambda x: input(f'{x}\n>')

USERNAME = prompt('What should I call you?')
BOTNAME = prompt('What is my name?')

userInput = ''
print('\nPress \'q\' to quit any time.\n')

while userInput != 'q':
    userInput = prompt('Listening...')