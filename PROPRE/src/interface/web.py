import gradio as gr
import time
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import process as process
from postprocessing.periodic_ollama_run import start_periodic_ollama_check


# MODEL = "mistral"  # Model name for Ollama
MODEL = "llama3.2"  # Model name for Ollama

SUBTITLES_FILES = {}


def process_video_pipeline(video, spellchecking, llm, radio_model, checkboxes_langages, progress, subs):
    """Orchestrates the video processing steps."""
    if video is None:
        return None
    
    progress(progress=0, desc="Processing video...")
    start_time = time.time()
    # Process the video
    output_video_path, output_subtitle_path = process.process(
        video, 
        use_spellchecking=spellchecking, 
        use_ollama_correct=llm, 
        model=MODEL, 
        translation_model=radio_model,
        translation_languages=checkboxes_langages,
        progress_component=progress,
    )
    # output_video_path, output_subtitle_path, message = None, None, None  # Mocked for now
    progress(progress=1, desc="Processing complete")
    end_time = time.time()
    time.sleep(0.05)
    elapsed_time = end_time - start_time
    
    global SUBTITLES_FILES
    SUBTITLES_FILES = {
        'English': output_subtitle_path + "_en.srt" if "English" in checkboxes_langages else None,
        'French': output_subtitle_path + "_fr.srt" if "French" in checkboxes_langages else None,
        'Spanish': output_subtitle_path + "_es.srt" if "Spanish" in checkboxes_langages else None,
    }
    
    gr.Info("Processed video in {:.2f} seconds".format(elapsed_time), title = "Success", duration=4)
    
    selected_subtitle_file = SUBTITLES_FILES.get(subs, None)
    if selected_subtitle_file is None:
        for lang, file in SUBTITLES_FILES.items():
            if file is not None:
                selected_subtitle_file = file
                break
    
    return (output_video_path, selected_subtitle_file)


def change_video_subs(video, subs):
    """Change the video subtitles based on the selected language."""
    if video is None:
        return None
    
    # Get the path to the selected subtitle file
    subtitle_file = SUBTITLES_FILES.get(subs, None)
    
    # If no subtitle file is found, return the video with the first available subtitle file    
    if subtitle_file is None:
        for lang, file in SUBTITLES_FILES.items():
            if file is not None:
                subtitle_file = file
                break
        # If no subtitle file is found, return the video without subtitles
        return video
    
    return video, subtitle_file


def update_available_languages(output_video_subs):
    available_languages = [lang for lang in SUBTITLES_FILES.keys() if SUBTITLES_FILES[lang] is not None]
    
    # If selected language is not available, default to the first available language
    selected_language = output_video_subs
    if output_video_subs not in available_languages:
        selected_language = available_languages[0] if available_languages else "English"
    
    return gr.update(choices=available_languages, value=selected_language)


def validate_language_selection(languages):
    """Ensure at least one language is selected."""
    if not languages:
        return gr.update(value=["English"])  # Default to English if none are selected
    return gr.update()


def validate_model_selection(model, languages):
    """Update available languages based on the selected model."""
    if model == "Custom":
        if "Spanish" in languages:
            languages.remove("Spanish")
        return gr.update(choices=["English", "French"], value=languages)
    elif model == "Opus-MT":
        return gr.update(choices=["English", "French", "Spanish"], value=languages)
    return gr.update()


if __name__ == "__main__":
    # Load CSS for styling
    with open("src/interface/style.css", "r") as f:
        css = f.read()

    # Default
    # Ocean
    with gr.Blocks(theme=gr.themes.Ocean(), title="Video Subtitling Pipeline", css=css) as demo: #primary_hue="purple", secondary_hue="blue"
        gr.Markdown("# Video Subtitling Pipeline")
        gr.Markdown("Upload a video to generate subtitles (closed captions) and add them to the video.")

        with gr.Row():
            with gr.Column():
                input_video = gr.Video(label="Upload Video")
                
                gr.Markdown("### Processing options")
                with gr.Column():
                    checkbox_spellchecking = gr.Checkbox(info="Types of correction", label="Correct transcription with spellchecking", value=True, interactive=True)
                    checkbox_llm = gr.Checkbox(label="Correct transcription with LLM", value=False, interactive=True)
                    with gr.Row():
                        radio_model = gr.Radio(show_label=False, info="Translation model", choices=["Custom", "Opus-MT"], value="Custom", interactive=True)
                        checkboxes_langages = gr.CheckboxGroup(show_label=False, info="Languages", choices=["English", "French"], value=["English"], interactive=True)
                    # gr.Markdown("### Translation")
                    
            with gr.Column():
                progress = gr.Progress()
                output_video = gr.Video(label="Subtitled Video", interactive=False, show_download_button=True, visible=True)
                gr.Markdown("### Subtitles Language")
                output_video_subs = gr.Radio(
                    choices=["English", "French", "Spanish"],
                    value="English",
                    interactive=True,
                    show_label=False,
                )
        
        input_video.change(
            fn=lambda video, spellchecking, llm, model_translate, langages, subs: process_video_pipeline(video, spellchecking, llm, model_translate, langages, progress, subs),
            inputs=[input_video, checkbox_spellchecking, checkbox_llm, radio_model, checkboxes_langages, output_video_subs],
            outputs=[output_video]
        )
        
        # When output video subs is changed, update the output video
        output_video_subs.change(
            show_progress='hidden',
            fn=change_video_subs,
            inputs=[input_video, output_video_subs],
            outputs=[output_video]
        )
        
        # When output video is changed, update the output video available languages
        output_video.change(
            show_progress='hidden',
            fn=update_available_languages,
            inputs=[output_video_subs],
            outputs=[output_video_subs],
        )
        
        # Add a change event to validate the selection
        checkboxes_langages.change(
            show_progress='hidden',
            fn=validate_language_selection,
            inputs=[checkboxes_langages],
            outputs=[checkboxes_langages]
        )
        
        # Add a change event to update the available languages based on the selected model
        radio_model.change(
            show_progress='hidden',
            fn=validate_model_selection,
            inputs=[radio_model, checkboxes_langages],
            outputs=[checkboxes_langages]
        )
        
    start_periodic_ollama_check(MODEL)  # Start the periodic Ollama check in a separate thread
    demo.launch(share=False)