from pydantic import BaseModel
import ollama
import os


# Load the prompt from a file
def _load_prompt(filename):
    global PROMPT
    with open(filename, 'r') as file:
        PROMPT = file.read()

PROMPT = None
_load_prompt(os.path.dirname(os.path.abspath(__file__)) + "/ollama-prompt.txt")
assert PROMPT, 'Prompt is empty'


# Define the schema for the response
class ProcessedSentence(BaseModel):
  processed_sentence: str


def correct_sentence(sentence: str, model: str = 'mistral') -> str:
    """
    Correct the given sentence using a Ollama model.\n
    Ex:\n
        Note: sentence can be a string or a list of strings.\n
        sentence = "some asid is often aded two inhibit climerisation in the twobe. if you want a sped up the seting of siperlu, one way is to had more negedtof ions, the initiators that start the plemerisation reaction. you can by excelerators spesificaly for this burpuse of the shelf."\n
    Output:\n
        "Some acid is often added to inhibit climerisation in the tube. If you want to speed up the setting of siperlu, one way is to add more negedtof ions, the initiators that start the polymerisation reaction. You can buy accelerators specifically for this purpose off the shelf."
    """
    global PROMPT
    if isinstance(sentence, list):
        sentence = ' '.join(sentence)
    elif not isinstance(sentence, str):
        raise ValueError("Input sentence must be a string or a list of strings.")
    temp_prompt = PROMPT.replace('{{ASR_output}}', sentence)
    
    # Call the Ollama API
    response = ollama.chat(
        model=model, 
        messages=[
            {
            'role': 'user',
            'content': temp_prompt,
            },
        ],
        format=ProcessedSentence.model_json_schema(),
    )

    # Use Pydantic to validate the response
    response = ProcessedSentence.model_validate_json(response.message.content)
    # print(response)
    return response.processed_sentence


if __name__ == "__main__":
    model = 'mistral'
    # model = 'mixtral'
    # model = 'llama3.2'
    # model = 'gemma3:1b'
    # model = 'deepseek-r1:14b'
    # model = 'llama3.1'
    
    sentence = "some asid is often aded two inhibit climerisation in the twobe. if you want a sped up the seting of siperlu, one way is to had more negedtof ions, the initiators that start the plemerisation reaction. you can by excelerators spesificaly for this burpuse of the shelf."
    corrected = correct_sentence(sentence, model=model)
    print("Input sentence:")
    print(sentence)
    print("\nCorrected sentence:")
    print(corrected)