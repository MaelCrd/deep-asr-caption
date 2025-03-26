from pydantic import BaseModel
import ollama

model = 'mistral'
# model = 'mixtral'
# model = 'llama3.2'
# model = 'gemma3:1b'

# Define the schema for the response
class ProcessedSentence(BaseModel):
  processed_sentence: str

# ASR_output = "annd d  the ttopthesuceserrrantt oofff  the  goddem  mie,   a sstolld  bhee  senn two bbbrak sttons,   rrronnd wwhichhh  thhe  watiss   hhagonnn sffilllly   a  verrddaatt ssosset,  annd  tthesssttinssa  sto  cod bbyrr thheppeet   off  the  ay."
# ASR_output = "annd tthen  ssaid   hhe  woned a bassterr  pplllaced  unndderrr   hamm   aasshe  wasss    un comfboann ann ggrraad  pann,  the docctorrn vvalla llifttedd  thhe  clokk  withhh whittchhhe  wass  coverr anns  mmakkinngg  rrri  faccesss    aat tthe  nnoteommessme   off  mmorredtefinng  fflusshh."
# ASR_output = "sshhe llligkess toorr toon thatt nnie  tooa  o llekehonn,  sotay  annd  semmplelasit wass    annnd   ffellllt  arrraatt   rrepotimannssttak sept,   aann y  pplllaise  wessshowwo  ne me  o wwith  femoyef fancagga."

# ASR_output = "wwhicchh  eat  wass   nevverrr   in diffferrnn  to tthe  ccrrredddit  off  doinng  evveryyythhinng  wwellllll  inn  attttenntively  witth  the  rrell  goood wwilll   off  a  mmiin  a llited wwitthh  itss ownn  iddeasss, did sheethenn  do  allll the  honeurrss  oot  the mealll."

while True:
    ASR_output = str(input("Enter the ASR output: "))

    with open('ollama-prompt.txt', 'r') as file:
        prompt = file.read()
    prompt = prompt.replace('{{ASR_output}}', ASR_output)

    # print("Prompt: ")
    # print(prompt)
    # quit()

    response = ollama.chat(
        model=model, 
        messages=[
            {
            'role': 'user',
            'content': prompt,
            },
        ],
        format=ProcessedSentence.model_json_schema(),
    )

    # Use Pydantic to validate the response
    response = ProcessedSentence.model_validate_json(response.message.content)
    # print(response)

    print()
    print("Initial   >>", ASR_output)
    print("Corrected >>", response.processed_sentence)

