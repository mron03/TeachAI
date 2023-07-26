import json

a = '''
    {
        "Write the topic name": {
            "Instruction 1": "Write What to do",
            "Speech 1": "Write what to tell for instruction 1",
            "Instruction 2": "Write What to do",
            "Speech 2": "Write what to tell for instruction 2",
            "Instruction 3": "Write What to do",
            "Speech 3": "Write what to tell for instruction 3"
        }
    }
'''
b = json.loads(a)
print(b)
a = [b]

for i in range(len(a)):

    for response in a:
        print(response)
        
        for topic, value in response.items():
            print(topic)
                
            for inst_speech, content in value.items():
                print(f'{inst_speech} : {content}')
                print()

            
            print()