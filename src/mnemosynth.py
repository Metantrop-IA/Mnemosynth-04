#!/usr/bin/env python3
"""
Mnemosynth: The Memory Synthesizer
A unified script combining TTS inference utilities and Gradio interface
"""

# ============================================================================
# IMPORTS AND INITIAL SETUP
# ============================================================================

import os
import sys
import re
import tempfile
import hashlib
import click
# from importlib.resources import files  # No longer needed

# Add current directory and parent directories to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add third party BigVGAN to path if it exists
bigvgan_path = os.path.join(project_root, "third_party", "BigVGAN")
if os.path.exists(bigvgan_path) and bigvgan_path not in sys.path:
    sys.path.append(bigvgan_path)

# Core libraries
import numpy as np
import torch
import torchaudio
import soundfile as sf
import tqdm
from cached_path import cached_path

# Audio processing
from pydub import AudioSegment, silence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

# Transformers and NLP
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from num2words import num2words
from sentence_transformers import SentenceTransformer

# RAG system
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import pdfplumber

# TTS models
from vocos import Vocos
from F5_TTS_Files import CFM, DiT, UNetT
from F5_TTS_Files.utils import (
    get_tokenizer,
    convert_char_to_pinyin,
)

# Gradio interface
import gradio as gr

# Spaces GPU decorator (if available)
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

# ============================================================================
# GLOBAL CONSTANTS AND CONFIGURATION
# ============================================================================

# Audio processing constants
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"
target_rms = 0.1
cross_fade_duration = 0.15
ode_method = "euler"
nfe_step = 32  # 16, 32
cfg_strength = 2.0
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None

# Device configuration
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Global state variables
_ref_audio_cache = {}
chat_model_state = None
chat_tokenizer_state = None

# ============================================================================
# UTILITY FUNCTIONS FROM utils_infer.py
# ============================================================================

def chunk_text(text, max_chars=135):
    """
    Splits the input text into chunks, each with a maximum number of characters.

    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def load_vocoder(vocoder_name="vocos", is_local=False, local_path="", device=device):
    """Load vocoder model"""
    if vocoder_name == "vocos":
        if is_local:
            print(f"Load vocos from local path {local_path}")
            vocoder = Vocos.from_hparams(f"{local_path}/config.yaml")
            state_dict = torch.load(f"{local_path}/pytorch_model.bin", map_location="cpu")
            vocoder.load_state_dict(state_dict)
            vocoder = vocoder.eval().to(device)
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
    elif vocoder_name == "bigvgan":
        try:
            from third_party.BigVGAN import bigvgan
        except ImportError:
            print("You need to follow the README to init submodule and change the BigVGAN source code.")
        if is_local:
            """download from https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x/tree/main"""
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)
        else:
            vocoder = bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False)

        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
    return vocoder


def load_checkpoint(model, ckpt_path, device, dtype=None, use_ema=True):
    """Load model checkpoint"""
    if dtype is None:
        dtype = (
            torch.float16 if device == "cuda" and torch.cuda.get_device_properties(device).major >= 6 else torch.float32
        )
    model = model.to(dtype)

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file
        checkpoint = load_file(ckpt_path)
    else:
        checkpoint = torch.load(ckpt_path, weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }

        # patch for backward compatibility, 305e3ea
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"])

    return model.to(device)


def load_model(
    model_cls,
    model_cfg,
    ckpt_path,
    mel_spec_type=mel_spec_type,
    vocab_file="",
    ode_method=ode_method,
    use_ema=True,
    device=device,
):
    """Load F5-TTS model"""
    if vocab_file == "":
        vocab_file = os.path.join(os.path.dirname(__file__), "F5_TTS_Files", "vocab.txt")
    tokenizer = "custom"

    print("\nvocab : ", vocab_file)
    print("tokenizer : ", tokenizer)
    print("model : ", ckpt_path, "\n")

    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    return model


def remove_silence_edges(audio, silence_threshold=-42):
    """Remove silence from audio edges"""
    # Remove silence from the start
    non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
    audio = audio[non_silent_start_idx:]

    # Remove silence from the end
    non_silent_end_duration = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        non_silent_end_duration -= 0.001
    trimmed_audio = audio[: int(non_silent_end_duration * 1000)]

    return trimmed_audio


def preprocess_ref_audio_text(ref_audio_orig, ref_text, clip_short=True, show_info=print, device=device):
    """Preprocess reference audio and text"""
    show_info("Converting audio...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio_orig)

        if clip_short:
            # 1. try to find long silence for clipping
            non_silent_segs = silence.split_on_silence(
                aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
                    show_info("Audio is over 15s, clipping short. (1)")
                    break
                non_silent_wave += non_silent_seg

            # 2. try to find short silence for clipping if 1. failed
            if len(non_silent_wave) > 15000:
                non_silent_segs = silence.split_on_silence(
                    aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
                )
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
                        show_info("Audio is over 15s, clipping short. (2)")
                        break
                    non_silent_wave += non_silent_seg

            aseg = non_silent_wave

            # 3. if no proper silence found for clipping
            if len(aseg) > 15000:
                aseg = aseg[:15000]
                show_info("Audio is over 15s, clipping short. (3)")

        aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
        aseg.export(f.name, format="wav")
        ref_audio = f.name

    # Compute a hash of the reference audio file
    with open(ref_audio, "rb") as audio_file:
        audio_data = audio_file.read()
        audio_hash = hashlib.md5(audio_data).hexdigest()

    global _ref_audio_cache
    if audio_hash in _ref_audio_cache:
        # Use cached reference text
        show_info("Using cached reference text...")
        ref_text = _ref_audio_cache[audio_hash]
    else:
        if not ref_text.strip():
            global asr_pipe
            if asr_pipe is None:
                # Initialize ASR pipeline if not already done
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                asr_pipe = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-large-v3-turbo",
                    torch_dtype=dtype,
                    device=device,
                    generate_kwargs={"task": "transcribe", "language": "es", "forced_decoder_ids": None}
                )
            show_info("No reference text provided, transcribing reference audio...")
            ref_text = asr_pipe(
                ref_audio,
                chunk_length_s=30,
                batch_size=128,
                return_timestamps=False,
                generate_kwargs={"forced_decoder_ids": None}
            )["text"].strip()
            show_info("Finished transcription")
        else:
            show_info("Using custom reference text...")
        # Cache the transcribed text
        _ref_audio_cache[audio_hash] = ref_text

    # Ensure ref_text ends with a proper sentence-ending punctuation
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    return ref_audio, ref_text


def infer_process(
    ref_audio,
    ref_text,
    gen_text,
    model_obj,
    vocoder,
    mel_spec_type=mel_spec_type,
    show_info=print,
    progress=tqdm,
    target_rms=target_rms,
    cross_fade_duration=cross_fade_duration,
    nfe_step=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
    speed=speed,
    fix_duration=fix_duration,
    device=device,
):
    """Main inference process"""
    # Split the input text into batches
    audio, sr = torchaudio.load(ref_audio)
    max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr) * (25 - audio.shape[-1] / sr))
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)
    for i, gen_text in enumerate(gen_text_batches):
        print(f"gen_text {i}", gen_text)

    show_info(f"Generating audio in {len(gen_text_batches)} batches...")
    return infer_batch_process(
        (audio, sr),
        ref_text,
        gen_text_batches,
        model_obj,
        vocoder,
        mel_spec_type=mel_spec_type,
        progress=progress,
        target_rms=target_rms,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        speed=speed,
        fix_duration=fix_duration,
        device=device,
    )


def infer_batch_process(
    ref_audio,
    ref_text,
    gen_text_batches,
    model_obj,
    vocoder,
    mel_spec_type="vocos",
    progress=tqdm,
    target_rms=0.1,
    cross_fade_duration=0.15,
    nfe_step=32,
    cfg_strength=2.0,
    sway_sampling_coef=-1,
    speed=1,
    fix_duration=None,
    device=None,
):
    """Process inference in batches"""
    audio, sr = ref_audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    generated_waves = []
    spectrograms = []

    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "
    for i, gen_text in enumerate(progress.tqdm(gen_text_batches)):
        # Prepare the text
        text_list = [ref_text + gen_text]
        final_text_list = convert_char_to_pinyin(text_list)

        ref_audio_len = audio.shape[-1] // hop_length
        if fix_duration is not None:
            duration = int(fix_duration * target_sample_rate / hop_length)
        else:
            # Calculate duration
            ref_text_len = len(ref_text.encode("utf-8"))
            gen_text_len = len(gen_text.encode("utf-8"))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

        # inference
        with torch.inference_mode():
            generated, _ = model_obj.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )

            generated = generated.to(torch.float32)
            generated = generated[:, ref_audio_len:, :]
            generated_mel_spec = generated.permute(0, 2, 1)
            if mel_spec_type == "vocos":
                generated_wave = vocoder.decode(generated_mel_spec)
            elif mel_spec_type == "bigvgan":
                generated_wave = vocoder(generated_mel_spec)
            if rms < target_rms:
                generated_wave = generated_wave * rms / target_rms

            # wav -> numpy
            generated_wave = generated_wave.squeeze().cpu().numpy()

            generated_waves.append(generated_wave)
            spectrograms.append(generated_mel_spec[0].cpu().numpy())

    # Combine all generated waves with cross-fading
    if cross_fade_duration <= 0:
        # Simply concatenate
        final_wave = np.concatenate(generated_waves)
    else:
        final_wave = generated_waves[0]
        for i in range(1, len(generated_waves)):
            prev_wave = final_wave
            next_wave = generated_waves[i]

            # Calculate cross-fade samples, ensuring it does not exceed wave lengths
            cross_fade_samples = int(cross_fade_duration * target_sample_rate)
            cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

            if cross_fade_samples <= 0:
                # No overlap possible, concatenate
                final_wave = np.concatenate([prev_wave, next_wave])
                continue

            # Overlapping parts
            prev_overlap = prev_wave[-cross_fade_samples:]
            next_overlap = next_wave[:cross_fade_samples]

            # Fade out and fade in
            fade_out = np.linspace(1, 0, cross_fade_samples)
            fade_in = np.linspace(0, 1, cross_fade_samples)

            # Cross-faded overlap
            cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

            # Combine
            new_wave = np.concatenate(
                [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
            )

            final_wave = new_wave

    # Create a combined spectrogram
    combined_spectrogram = np.concatenate(spectrograms, axis=1)

    return final_wave, target_sample_rate, combined_spectrogram


def remove_silence_for_generated_wav(filename):
    """Remove silence from generated wav file"""
    aseg = AudioSegment.from_file(filename)
    non_silent_segs = silence.split_on_silence(
        aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500, seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        non_silent_wave += non_silent_seg
    aseg = non_silent_wave
    aseg.export(filename, format="wav")


def save_spectrogram(spectrogram, path):
    """Save spectrogram plot"""
    plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    plt.colorbar()
    plt.savefig(path)
    plt.close()

# ============================================================================
# GRADIO INTERFACE FUNCTIONS
# ============================================================================

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
    """Convert numbers to text in Spanish"""
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
    """Main TTS inference function for Gradio"""
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
    """Parse speech types from text"""
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

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

# Initialize ASR pipeline
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    torch_dtype=dtype,
    device=device,
    generate_kwargs={"task": "transcribe", "language": "es", "forced_decoder_ids": None}
)

# Load vocoder
vocoder = load_vocoder()

# Load F5-TTS model
F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
F5TTS_ema_model = load_model(
    DiT, F5TTS_model_cfg, str(cached_path("hf://jpgallegoar/F5-Spanish/model_1200000.safetensors"))
)

# ============================================================================
# GRADIO INTERFACE SETUP
# ============================================================================

def create_chat_interface():
    """Create the main chat interface"""
    
    # Initialize chat model
    global chat_model_state, chat_tokenizer_state
    if chat_model_state is None:
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        chat_model_state = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        chat_tokenizer_state = AutoTokenizer.from_pretrained(model_name)

    # Setup asset paths
    ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "Assets"))
    initial_prompt_path = os.path.join(ASSETS_DIR, "Initial_Prompt.txt")
    voice_ref_trans_path = os.path.join(ASSETS_DIR, "Voice_Ref_Trans.txt")
    voice_ref_wav_path = os.path.join(ASSETS_DIR, "Voice_Ref.wav")

    # Load initial prompt and reference files
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

    # RAG System Setup
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

    # Create Gradio interface
    with gr.Blocks() as app_chat:
        gr.Markdown(
            f"""
    # Mnemosynth
    The Memory Synthesizer of {read_personality_file()}
    """
        )

        # Hidden components for TTS configuration
        ref_audio_chat = gr.Audio(value=voice_ref_wav_path, visible=False, type="filepath")
        model_choice_chat = gr.Radio(choices=["F5-TTS"], label="Seleccionar Modelo TTS", value="F5-TTS", visible=False)
        remove_silence_chat = gr.Checkbox(value=True, visible=False)
        ref_text_chat = gr.Textbox(value=voice_ref_trans, visible=False)
        system_prompt_chat = gr.Textbox(value=initial_prompt, visible=False)

        # Main interface components
        with gr.Row():
            audio_input_chat = gr.Microphone(label="Record your message", type="filepath")
        with gr.Row():
            audio_output_chat = gr.Audio(autoplay=True, label="Answer")
        
        with gr.Row():
            text_input_chat = gr.Textbox(label="Write your message", lines=1)
        with gr.Row():        
            send_btn_chat = gr.Button("Send")
        with gr.Row():
            chatbot_interface = gr.Chatbot(
                label="Conversation",
                type="messages"
            )
        with gr.Row():
            clear_btn_chat = gr.Button("Clear")

        # Conversation state
        conversation_state = gr.State(
            value=[
                {
                    "role": "system",
                    "content": initial_prompt
                }
            ]
        )

        # Event handlers
        @gpu_decorator
        def process_audio_input(audio_path, text, history, conv_state):
            if not audio_path and not text.strip():
                return history, conv_state

            if audio_path:
                text = preprocess_ref_audio_text(audio_path, text)[1]

            if not text.strip():
                return history, conv_state

            # RAG: Retrieve relevant context
            contexto = " ".join(retrieve_context(text, top_k=3))
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
                    "content": initial_prompt
                }
            ]

        def update_system_prompt(new_prompt):
            """Update the system prompt and reset the conversation"""
            new_conv_state = [{"role": "system", "content": new_prompt}]
            return [], new_conv_state

        # Audio input handler
        audio_input_chat.stop_recording(
            fn=process_audio_input,
            inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
            outputs=[chatbot_interface, conversation_state]
        ).then(
            fn=generate_audio_response,
            inputs=[chatbot_interface, ref_audio_chat, ref_text_chat, model_choice_chat, remove_silence_chat],
            outputs=[audio_output_chat]
        )

        # Text input handler
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

        # Send button handler
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

        # Clear button handler
        clear_btn_chat.click(
            clear_conversation,
            outputs=[chatbot_interface, conversation_state],
        )

        # System prompt change handler
        system_prompt_chat.change(
            update_system_prompt,
            inputs=system_prompt_chat,
            outputs=[chatbot_interface, conversation_state],
        )

    return app_chat


def create_credits_interface():
    """Create the credits interface"""
    with gr.Blocks() as app_credits:
        gr.Markdown("""
    # Credits
    * [ΜΕΤΑΝΘΡΩΠΙΑ](https://github.com/METANTROP-IA) by this [Mnemosynth demo.](https://github.com/Metantrop-IA/Mnemosynth-02)
    * [OpenAI](https://huggingface.co/openai) by Whisper.
    * [Alibaba](https://huggingface.co/Qwen) by Qwen.                     
    * [Yushen Chen](https://huggingface.co/SWivid) by the original [F5-TTS paper.](https://arxiv.org/abs/2410.06885)
    * [jpgallegoar](https://github.com/jpgallegoar) by the Spanish fine-tuning of F5-TTS model.
                    
    """)
    return app_credits


def create_main_app():
    """Create the main tabbed application"""
    app_chat = create_chat_interface()
    app_credits = create_credits_interface()
    
    with gr.Blocks() as app:
        gr.TabbedInterface(
            [app_chat, app_credits],
            ["Mnemosynth", "Credits"],
        )
    
    return app

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

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
    """Main function to launch the Mnemosynth application"""
    print("Iniciando Mnemosynth - The Memory Synthesizer...")
    app = create_main_app()
    app.queue(api_open=api).launch(
        server_name=host, 
        server_port=port, 
        share=share, 
        show_api=api
    )


if __name__ == "__main__":
    if not USING_SPACES:
        main()
    else:
        app = create_main_app()
        app.queue().launch()
