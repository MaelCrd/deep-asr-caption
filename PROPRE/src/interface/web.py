import gradio as gr
import time
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import process as process


MESSAGE = ""


def process_video_pipeline(video):
    """Orchestrates the video processing steps."""
    global MESSAGE
    if video is None:
        MESSAGE = ""
        return None
    # print(video)
    
    start_time = time.time()
    # Process the video
    output_video_path, output_subtitle_path, message = process.process(video)  # Call the process function from your module
    end_time = time.time()
    elapsed_time = end_time - start_time
    MESSAGE = message
    
    
    gr.Info("Processed video in {:.2f} seconds".format(elapsed_time), title = "Success", duration=5)
    
    return output_video_path, output_subtitle_path


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
"""

# Default
# Ocean
with gr.Blocks(theme=gr.themes.Ocean(), title="Video Subtitling Pipeline", css=css) as demo: #primary_hue="purple", secondary_hue="blue"
    gr.Markdown("# Video Subtitling Pipeline")
    gr.Markdown("Upload a video to generate subtitles and add them to the video.")

    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Upload Video")
            # predicted_output = gr.Textbox(label="Step 1: Predicted Transcription", interactive=False)
            # processed_output = gr.Textbox(label="Step 2: Post-Processed Transcription", interactive=False)
        with gr.Column():
            output_video = gr.Video(label="Subtitled Video", interactive=False)
            # output_text = gr.Textbox(label="Processing information", interactive=False, visible=False)
            # number = gr.Number(label="Runtime", interactive=False, visible=True, value=1.0)
            # output_video = gr.Video(('data/output/videos/youtube_1B3B_LLM-26s_subs.mp4', 'data/output/subtitles/youtube_1B3B_LLM-26s.srt'), label="Test", interactive=False)

    # def update_end_process(video_path):
    #     output_video_path, output_subtitle_path, message = process.process(video_path)  # Call the process function from your module
    #     return output_video_path, output_subtitle_path, gr.update(output_text, visible=True, value=message)
    
    def update_message(_arg):
        return gr.update(visible=len(MESSAGE) > 0, value=MESSAGE)
    
    input_video.change(
        fn=process_video_pipeline,
        inputs=input_video,
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
    demo.launch()