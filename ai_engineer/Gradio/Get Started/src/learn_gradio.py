import gradio as gr

def process_input(input_text):
    # Process the input text using Gradio
    return "Process the text: " + input_text

demo = gr.Interface(fn=process_input, inputs="text", outputs="text")
demo.launch()
