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


def correct_sentence(sentence: str, model: str = 'llama3.2', progress_callback=None) -> str:
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
    
    # Split the sentence into chunks if it's too long
    if len(sentence) > 1000:
        sentences = sentence.split('.')
        # Re-add the periods to the sentences
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        chunks = []
        current_chunk = ""
        for s in sentences:
            if len(current_chunk) + len(s) + 1 <= 1000:
                current_chunk += s + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = s + " "
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())
    else:
        chunks = [sentence]
    
    # print(f"Input sentence: {sentence}")
    # print(chunks)
    
    full_corrected_sentence = ""
    # Process each chunk separately
    for i in range(len(chunks)):
        temp_prompt = PROMPT.replace('{{ASR_output}}', chunks[i])
        # print(f"Prompt: {temp_prompt}")
        
        # Call the Ollama API with streaming enabled
        response_text = ""
        for part in ollama.chat(
            model=model, 
            messages=[
                {
                'role': 'user',
                'content': temp_prompt,
                },
            ],
            format=ProcessedSentence.model_json_schema(),
            stream=True,  # Enable streaming
        ):
            # Append the streamed content
            response_text += part['message']['content']
            # Call the progress callback if provided
            if progress_callback:
                progress_callback((len(full_corrected_sentence) + len(response_text) - 65) / len(sentence))

        # print(f"Response: {response_text}")

        # Use Pydantic to validate the response
        response = ProcessedSentence.model_validate_json(response_text)
        
        # Add the processed sentence to the full corrected sentence
        if i == 0:
            full_corrected_sentence += response.processed_sentence
        else:
            full_corrected_sentence += " " + response.processed_sentence
        
        # print(response)
    return full_corrected_sentence


if __name__ == "__main__":
    model = 'mistral' # Bien mais un peu lent
    # model = 'mixtral'
    model = 'llama3.2' # Bien et vitesse ok
    # model = 'gemma3:1b'
    # model = 'deepseek-r1:14b'
    # model = 'llama3.1'
    
    sentence = "some asid is often aded two inhibit climerisation in the twobe. if you want a sped up the seting of siperlu, one way is to had more negedtof ions, the initiators that start the plemerisation reaction. you can by excelerators spesificaly for this burpuse of the shelf."
    corrected = correct_sentence(sentence, model=model)
    print("Input sentence:")
    print(sentence)
    print("\nCorrected sentence:")
    print(corrected)