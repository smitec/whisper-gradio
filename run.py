import gradio as gr
import whisper

def speech_to_text(tmp_filename, model_size):
    model = whisper.load_model(model_size)
    result = model.transcribe(tmp_filename)

    return result["text"]


gr.Interface(
    fn=speech_to_text,
    inputs=[
        gr.Audio(source="microphone", type="filepath"),
        gr.Dropdown(choices=["tiny", "base", "small", "medium", "large"]),
        ],
    outputs="text").launch()
