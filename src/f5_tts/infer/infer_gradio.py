import re
import tempfile
import os

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
from cached_path import cached_path
from transformers import AutoModelForCausalLM, AutoTokenizer
from num2words import num2words
from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import pdfplumber

try:
    import spaces

    USING_SPACES = True
except ImportError:
    USING_SPACES = False


def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func

def read_personality_file():
    """Read the content of the Personality.txt file"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        personality_path = os.path.join(current_dir, "Assets", "Personality.txt")
        with open(personality_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            print(f"Personality.txt cargado exitosamente desde: {personality_path}")
            return content
    except FileNotFoundError:
        error_msg = "Error: No se pudo encontrar el archivo Personality.txt"
        print(error_msg)
        return error_msg


from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)

vocoder = load_vocoder()

# load models
F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
F5TTS_ema_model = load_model(
    DiT, F5TTS_model_cfg, str(cached_path("hf://jpgallegoar/F5-Spanish/model_1200000.safetensors"))
)

chat_model_state = None
chat_tokenizer_state = None

@gpu_decorator
def generate_response(messages, model, tokenizer):
    """Generate response using Qwen"""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
    )

    generated_ids = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def traducir_numero_a_texto(texto):
    texto_separado = re.sub(r'([A-Za-z])(\d)', r'\1 \2', texto)
    texto_separado = re.sub(r'(\d)([A-Za-z])', r'\1 \2', texto_separado)
    
    def reemplazar_numero(match):
        numero = match.group()
        return num2words(int(numero), lang='es')

    texto_traducido = re.sub(r'\b\d+\b', reemplazar_numero, texto_separado)

    return texto_traducido

@gpu_decorator
def infer(
    ref_audio_orig, ref_text, gen_text, model, remove_silence, cross_fade_duration=0.15, speed=1, show_info=gr.Info
):
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)

    ema_model = F5TTS_ema_model

    if not gen_text.startswith(" "):
        gen_text = " " + gen_text
    if not gen_text.endswith(". "):
        gen_text += ". "

    gen_text = gen_text.lower()
    gen_text = traducir_numero_a_texto(gen_text)

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        speed=speed,
        show_info=show_info,
        progress=gr.Progress(),
    )

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Save the spectrogram
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave), spectrogram_path


def parse_speechtypes_text(gen_text):
    # Pattern to find {speechtype}
    pattern = r"\{(.*?)\}"

    # Split the text by the pattern
    tokens = re.split(pattern, gen_text)

    segments = []

    current_style = "Regular"

    for i in range(len(tokens)):
        if i % 2 == 0:
            # This is text
            text = tokens[i].strip()
            if text:
                segments.append({"style": current_style, "text": text})
        else:
            # This is style
            style = tokens[i].strip()
            current_style = style

    return segments


########## INICIA GRADIO PRINCIPAL ##########
with gr.Blocks() as app_chat:
    gr.Markdown(
        f"""
# Mnemosynth
The Memory Synthesizer {read_personality_file()}
"""
    )

    chat_interface_container = gr.Column()

    if chat_model_state is None:
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        chat_model_state = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        chat_tokenizer_state = AutoTokenizer.from_pretrained(model_name)

    # Construir rutas absolutas a los assets
    ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "Assets"))
    initial_prompt_path = os.path.join(ASSETS_DIR, "Initial_Prompt.txt")
    voice_ref_trans_path = os.path.join(ASSETS_DIR, "Voice_Ref_Trans.txt")
    docs_RAG_path = os.path.join(ASSETS_DIR, "Docs_RAG.txt")
    voice_ref_wav_path = os.path.join(ASSETS_DIR, "Voice_Ref.wav")

    # Cargar contenido del prompt inicial y archivos
    try:
        with open(initial_prompt_path, "r", encoding="utf-8") as f:
            initial_prompt = f.read().strip()
            print(f"Archivo Initial_Prompt.txt cargado exitosamente en: {initial_prompt_path}")
    except Exception as e:
        print(f"Error leyendo Initial_Prompt.txt en {initial_prompt_path}: {e}")
        initial_prompt = "No eres un asistente de IA, eres quien el usuario diga que eres..."

    try:
        with open(voice_ref_trans_path, "r") as f:
            voice_ref_trans = f.read().strip()
            print(f"Archivo Voice_Ref_Trans.txt cargado exitosamente en: {voice_ref_trans_path}")
    except Exception as e:
        print(f"Error leyendo archivo Voice_Ref_Trans.txt en {voice_ref_trans_path}: {e}")
        voice_ref_trans = ""

  # --- RAG SYSTEM: Load and index PDFs from /src/f5_tts/infer/Assets/RAG ---
    PDF_RAG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "Assets", "RAG"))
    if os.path.isdir(PDF_RAG_DIR):
        pdf_files = [os.path.join(PDF_RAG_DIR, f) for f in os.listdir(PDF_RAG_DIR) if f.lower().endswith('.pdf')]
    else:
        print(f"ADVERTENCIA: Carpeta de PDFs para RAG no encontrada: {PDF_RAG_DIR}")
        pdf_files = []

    rag_documents = []
    if pdf_files:
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(pdf_file)
                rag_documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading PDF {pdf_file}: {e}")
    else:
        print("No PDF files found for RAG context retrieval.")

    if not rag_documents:
        print("ADVERTENCIA: No se encontraron PDFs en Assets/RAG. La recuperación de contexto estará deshabilitada.")
        chroma_db = None
        def retrieve_context(query, top_k=3):
            return []
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        rag_chunks = splitter.split_documents(rag_documents)
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        chroma_db = Chroma.from_documents(rag_chunks, embedding, persist_directory=os.path.join(PDF_RAG_DIR, "chroma_db"))
        def retrieve_context(query, top_k=3):
            results = chroma_db.similarity_search(query, k=top_k)
            return [doc.page_content for doc in results]

    # Asignar rutas y valores usando componentes Gradio
    ref_audio_chat = gr.Audio(value=voice_ref_wav_path, visible=False, type="filepath")
    model_choice_chat = gr.Radio(choices=["F5-TTS"], label="Seleccionar Modelo TTS", value="F5-TTS", visible=False)
    remove_silence_chat = gr.Checkbox(value=True, visible=False)
    ref_text_chat = gr.Textbox(value=voice_ref_trans, visible=False)
    system_prompt_chat = gr.Textbox(value=initial_prompt, visible=False)    #Crea la interfaz de Gradio 
    
    ### GRADIO INTERFACE START ###
    with gr.Row():
        audio_input_chat = gr.Microphone(label="Record you message",type="filepath")
    with gr.Row():
        audio_output_chat = gr.Audio(autoplay=True, label="Answer")
    
    with gr.Row():
        text_input_chat = gr.Textbox(label="Write your Message",lines=1)
    with gr.Row():        
        send_btn_chat = gr.Button("Send")
    with gr.Row():
        chatbot_interface = gr.Chatbot(
            label="Conversation",
            type="messages"
        )
    with gr.Row():
        clear_btn_chat = gr.Button("Clear")
    ### GRADIO INTERFACE STOP ###

    conversation_state = gr.State(
        value=[
            {
                "role": "system",
                "content": initial_prompt
            }
        ]
    )


    @gpu_decorator
    def process_audio_input(audio_path, text, history, conv_state):
        if not audio_path and not text.strip():
            return history, conv_state

        if audio_path:
            text = preprocess_ref_audio_text(audio_path, text)[1]

        if not text.strip():
            return history, conv_state

        # --- RAG: Recupera contexto relevante ---
        contexto = " ".join(retrieve_context(text, top_k=3))
        # Puedes incluir el contexto como parte del mensaje del sistema o del usuario
        conv_state.append({"role": "system", "content": f"Contexto relevante: {contexto}"})
        conv_state.append({"role": "user", "content": text})

        response = generate_response(conv_state, chat_model_state, chat_tokenizer_state)
        conv_state.append({"role": "assistant", "content": response})

        if not history:
            history = []
        history.extend([
            {"role": "user", "content": text},
            {"role": "assistant", "content": response}
        ])

        return history, conv_state

    @gpu_decorator
    def generate_audio_response(history, ref_audio, ref_text, model, remove_silence):
        """Generate TTS audio for AI response"""
        if not history or not ref_audio:
            return None

        # Get last assistant message
        last_message = next((msg for msg in reversed(history) 
                           if msg["role"] == "assistant"), None)
        if not last_message:
            return None

        audio_result, _ = infer(
            ref_audio,
            ref_text,
            last_message["content"],
            model,
            remove_silence,
            cross_fade_duration=0.15,
            speed=1.0,
            show_info=print
        )
        return audio_result

    def clear_conversation():
        """Reset the conversation"""
        return [], [
            {
                "role": "system",
                "content": initial_prompt  # Usar el prompt del archivo
            }
        ]

    def update_system_prompt(new_prompt):
        """Update the system prompt and reset the conversation"""
        new_conv_state = [{"role": "system", "content": new_prompt}]
        return [], new_conv_state

    # Handle audio input
    audio_input_chat.stop_recording(
        fn=process_audio_input,
        inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
        outputs=[chatbot_interface, conversation_state]
    ).then(
        fn=generate_audio_response,
        inputs=[chatbot_interface, ref_audio_chat, ref_text_chat, model_choice_chat, remove_silence_chat],
        outputs=[audio_output_chat]
    )

    # Handle text input
    text_input_chat.submit(
        process_audio_input,
        inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
        outputs=[chatbot_interface, conversation_state],
    ).then(
        generate_audio_response,
        inputs=[chatbot_interface, ref_audio_chat, ref_text_chat, model_choice_chat, remove_silence_chat],
        outputs=[audio_output_chat],
    ).then(
        lambda: None,
        None,
        text_input_chat,
    )

    # Handle send button
    send_btn_chat.click(
        process_audio_input,
        inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
        outputs=[chatbot_interface, conversation_state],
    ).then(
        generate_audio_response,
        inputs=[chatbot_interface, ref_audio_chat, ref_text_chat, model_choice_chat, remove_silence_chat],
        outputs=[audio_output_chat],
    ).then(
        lambda: None,
        None,
        text_input_chat,
    )

    # Handle clear button
    clear_btn_chat.click(
        clear_conversation,
        outputs=[chatbot_interface, conversation_state],
    )

    # Handle system prompt change and reset conversation
    system_prompt_chat.change(
        update_system_prompt,
        inputs=system_prompt_chat,
        outputs=[chatbot_interface, conversation_state],
    )
########## TERMINA GRADIO PRINCIPAL ##########

with gr.Blocks() as app_credits:
    gr.Markdown("""
# Credits
* [ΜΕΤΑΝΘΡΩΠΙΑ](https://github.com/METANTROP-IA) by the [Mnemosynth demo.](https://github.com/Metantrop-IA/Mnemosynth-01)
* [OpenAI](https://huggingface.co/openai) by Whisper.
* [Alibaba](https://huggingface.co/Qwen) by Qwen.                     
* [Yushen Chen](https://huggingface.co/SWivid) by the original [F5-TTS paper.](https://arxiv.org/abs/2410.06885)
* [jpgallegoar](https://github.com/jpgallegoar) by the Spanish fine-tuning of F5-TTS model.
                
""")

with gr.Blocks() as app:

    gr.TabbedInterface(
        [app_chat, app_credits],
        ["Mnemosynth", "Credits"],
    )

@click.command()
@click.option("--port", "-p", default=None, type=int, help="Puerto para ejecutar la aplicación")
@click.option("--host", "-H", default=None, help="Host para ejecutar la aplicación")
@click.option(
    "--share",
    "-s",
    default=False,
    is_flag=True,
    help="Compartir la aplicación a través de un enlace compartido de Gradio",
)
@click.option("--api", "-a", default=True, is_flag=True, help="Permitir acceso a la API")
def main(port, host, share, api):
    global app
    print("Iniciando la aplicación...")
    app.queue(api_open=api).launch(server_name=host, server_port=port, share=True, show_api=api)


if __name__ == "__main__":
    if not USING_SPACES:
        main()
    else:
        app.queue().launch()
