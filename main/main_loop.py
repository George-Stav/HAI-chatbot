import os
from question import answer

prompt = lambda x: input(f'{x}> ')

USERNAME = prompt('What should I call you?\n')
# BOTNAME = prompt('What is my name?')

userInput = ''
print('\nEnter \'q\' to quit any time.')

while True:
    userInput = prompt('\nListening...')

    if userInput == '':
        continue
    elif userInput == 'q':
        break

    print(answer(userInput))

print("Bye I guess.")