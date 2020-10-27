
import json

fo = open('parsedMessages.txt', 'w')


def parse(chat):
    messages = chat['messages']
    for message in messages:
        if message['sender_name'] == 'Ray Yin':
            try:
                o = message['content'].encode('ascii', 'ignore').decode('utf-8')
                fo.write(o + '\n')
            except:
                pass


files = ['pm']
for i in range(1, 11):
    files.append('cdc' + str(i))
for f in files:
    with open('./data/' + f + '.json', 'r') as f:
        chat = json.load(f)
        parse(chat)
