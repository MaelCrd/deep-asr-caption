import gradio as gr
import time
import os
import sys
import threading

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import process as process
from postprocessing.periodic_ollama_run import start_periodic_ollama_check


# MODEL = "mistral"  # Model name for Ollama
MODEL = "llama3.2"  # Model name for Ollama


MESSAGE = ""
def process_video_pipeline(video, spellchecking, llm, progress):
    """Orchestrates the video processing steps."""
    global MESSAGE
    if video is None:
        MESSAGE = ""
        return None
    # print(video)
    
    progress(progress=0, desc="Processing video...")
    
    start_time = time.time()
    # Process the video
    output_video_path, output_subtitle_path, message = process.process(video, use_spellchecking=spellchecking, use_ollama_correct=llm, model=MODEL, progress_component=progress)
    # output_video_path, output_subtitle_path, message = None, None, None  # Mocked for now
    progress(progress=1, desc="Processing complete")
    end_time = time.time()
    time.sleep(0.05)
    elapsed_time = end_time - start_time
    MESSAGE = message
    
    gr.Info("Processed video in {:.2f} seconds".format(elapsed_time), title = "Success", duration=4)
    
    return (output_video_path, output_subtitle_path)


css = """
video::cue {
    font-size: 24px;
    border: 1px solid green;
}

video::-webkit-media-text-track-display {
    padding: 8px;
    padding-left: 70px;
    padding-right: 70px;
    line-height: 29.5px;
}

video::-webkit-media-text-track-container {
}




.dark .toast-icon.info {
    color: var(--color-green-500) !important;
}

.dark .toast-text.info {
    color: var(--color-grey-200) !important;
}

.dark .timer.info {
    background: var(--color-green-500) !important;
}

.dark .toast-body.info {
    border: 1px solid var(--color-green-500) !important;
}

"""

# .dark .toast-title.info {
#     color: var(--color-green-500) !important;
# }

# Default
# Ocean
with gr.Blocks(theme=gr.themes.Ocean(), title="Video Subtitling Pipeline", css=css) as demo: #primary_hue="purple", secondary_hue="blue"
    gr.Markdown("# Video Subtitling Pipeline")
    gr.Markdown("Upload a video to generate subtitles and add them to the video.")

    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Upload Video")
            # checkbox_postprocess = gr.Checkbox(label="Post-process", value=True, interactive=True)
            
            with gr.Column():
                gr.Markdown("### Processing Options")
                checkbox_spellchecking = gr.Checkbox(label="Correct transcription with spellchecking", value=True, interactive=True)
                checkbox_llm = gr.Checkbox(label="Correct transcription with LLM", value=False, interactive=True)
                # with gr.Row():
                #     min_chars = gr.Number(label="Min characters", value=20, interactive=True)
                #     max_chars = gr.Number(label="Max characters", value=60, interactive=True)
                # checkbox_2 = gr.Checkbox(label="Step 2: Post-Process Transcription", value=True, interactive=True)
            
            # predicted_output = gr.Textbox(label="Step 1: Predicted Transcription", interactive=False)
            # processed_output = gr.Textbox(label="Step 2: Post-Processed Transcription", interactive=False)
        with gr.Column():
            output_video = gr.Video(label="Subtitled Video", interactive=False, show_download_button=True, visible=True)
            progress = gr.Progress()
            # gr.DownloadButton(label="Download Subtitled Video", interactive=True, variant="primary")
            # output_text = gr.Textbox(label="Processing information", interactive=False, visible=False)
            # number = gr.Number(label="Runtime", interactive=False, visible=True, value=1.0)
            # output_video = gr.Video(('data/output/videos/youtube_1B3B_LLM-26s_subs.mp4', 'data/output/subtitles/youtube_1B3B_LLM-26s.srt'), label="Test", interactive=False)

    # def update_end_process(video_path):
    #     output_video_path, output_subtitle_path, message = process.process(video_path)  # Call the process function from your module
    #     return output_video_path, output_subtitle_path, gr.update(output_text, visible=True, value=message)
    
    def update_message(_arg):
        return gr.update(visible=len(MESSAGE) > 0, value=MESSAGE)
    
    # input_video.change(
    #     fn=process_video_pipeline,
    #     inputs=[input_video, checkbox_spellchecking, checkbox_llm, progress],
    #     outputs=[output_video]
    # )
    input_video.change(
        fn=lambda video, spellchecking, llm: process_video_pipeline(video, spellchecking, llm, progress),
        inputs=[input_video, checkbox_spellchecking, checkbox_llm],
        outputs=[output_video]
    )
    
    # output_video.change(
    #     fn=update_message,
    #     inputs=output_video,
    #     outputs=[output_text]
    # )
    
    # input_video.change(
    #     fn=update_end_process,
    #     inputs=input_video,
    #     outputs=[output_video, output_video, output_text],
    # )

if __name__ == "__main__":
    start_periodic_ollama_check(MODEL)  # Start the periodic Ollama check in a separate thread
    demo.launch(share=False)