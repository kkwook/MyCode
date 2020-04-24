''' list Comprehension 연습하기 '''

# sentences에서 stopwords 뺀 단어만 추출하기
sentences = ["a new world record was set",
             "in the holy city of ayodhya",
             "on the eve of diwali on tuesday",
             "with over three lakh diya or earthen lamps",
             "lit up simultaneously on the banks of the sarayu river"]

stopwords = ['for', 'a', 'of', 'the', 'and', 'to', 'in', 'on', 'with']

list_result = [words for sentence in sentences for words in sentence.split(" ") if words not in stopwords]
print(list_result)

list_result2 = [[words for words in sentence.split(" ") if words not in stopwords] for sentence in sentences]
print(list_result2)

listA = []
for sentence in sentences:
    listB = []
    for word in sentence.split(" "):
        if word not in stopwords:
            listB.append(word)
            print(listB)
    listA.append(listB)




print(listB)
for x in listA:
    print(x)
