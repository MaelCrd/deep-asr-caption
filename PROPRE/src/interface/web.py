import gradio as gr
import time
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import process as process
from postprocessing.periodic_ollama_run import start_periodic_ollama_check


# MODEL = "mistral"  # Model name for Ollama
MODEL = "llama3.2"  # Model name for Ollama


def process_video_pipeline(video, spellchecking, llm, progress):
    """Orchestrates the video processing steps."""
    if video is None:
        return None
    
    progress(progress=0, desc="Processing video...")
    start_time = time.time()
    # Process the video
    output_video_path, output_subtitle_path = process.process(video, use_spellchecking=spellchecking, use_ollama_correct=llm, model=MODEL, progress_component=progress)
    # output_video_path, output_subtitle_path, message = None, None, None  # Mocked for now
    progress(progress=1, desc="Processing complete")
    end_time = time.time()
    time.sleep(0.05)
    elapsed_time = end_time - start_time
    
    gr.Info("Processed video in {:.2f} seconds".format(elapsed_time), title = "Success", duration=4)
    
    return (output_video_path, output_subtitle_path)


if __name__ == "__main__":
    # Load CSS for styling
    with open("src/interface/style.css", "r") as f:
        css = f.read()

    # Default
    # Ocean
    with gr.Blocks(theme=gr.themes.Ocean(), title="Video Subtitling Pipeline", css=css) as demo: #primary_hue="purple", secondary_hue="blue"
        gr.Markdown("# Video Subtitling Pipeline")
        gr.Markdown("Upload a video to generate subtitles and add them to the video.")

        with gr.Row():
            with gr.Column():
                input_video = gr.Video(label="Upload Video")
                
                with gr.Column():
                    gr.Markdown("### Processing Options")
                    checkbox_spellchecking = gr.Checkbox(label="Correct transcription with spellchecking", value=True, interactive=True)
                    checkbox_llm = gr.Checkbox(label="Correct transcription with LLM", value=False, interactive=True)
                    
            with gr.Column():
                output_video = gr.Video(label="Subtitled Video", interactive=False, show_download_button=True, visible=True)
                progress = gr.Progress()
        
        input_video.change(
            fn=lambda video, spellchecking, llm: process_video_pipeline(video, spellchecking, llm, progress),
            inputs=[input_video, checkbox_spellchecking, checkbox_llm],
            outputs=[output_video]
        )
        
    start_periodic_ollama_check(MODEL)  # Start the periodic Ollama check in a separate thread
    demo.launch(share=False)