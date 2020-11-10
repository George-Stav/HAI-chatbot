import math

f = open('../data/small_talk/qna_chitchat_witty.qna', 'r')

questions = {x:[] for x in range(100)}
answers = []

index = 0

for x in f:
    if x[0] == '-':
        questions[math.floor(index/2)].append(x[2:-1])
    elif x[0] == '`':
        line = f.readline()
        if line != '':
            answers.append(line[4:-1])
            index += 1

answers = [x for x in answers if x]
questions = {x:questions[x] for x in questions if questions[x]}

f.close()

questions['answers'] = answers

print(questions)




# for (w, p) in zip(witty_q, prof_q):
#     if witty_q[w][0] != prof_q[p][0]:
#         print("w: " + str(w) + " ===> " + witty_q[w][0])
#         print("p: " + str(p) + " ===> " + prof_q[w][0])