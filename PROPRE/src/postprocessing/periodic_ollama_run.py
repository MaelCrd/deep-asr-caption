import time
import ollama
import threading

def _periodic_ollama_run(model):
    """
    Send a request to the Ollama API to keep the model loaded in memory.
    """
    response = ollama.chat(
        model=model,
        messages=[
            {
            'role': 'user',
            'content': 'Hey',
            },
        ],
    )
    print(response)


def start_periodic_ollama_check(model):
    """
    Starts a separate thread to periodically call the Ollama API.
    """
    def periodic_check():
        while True:
            try:
                print("Performing periodic Ollama check...")
                _periodic_ollama_run(model)
            except Exception as e:
                print(f"Error during periodic Ollama check: {e}")
            time.sleep(180)  # Wait for X minutes

    thread = threading.Thread(target=periodic_check, daemon=True)
    thread.start()