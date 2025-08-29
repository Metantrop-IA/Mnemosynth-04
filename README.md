# Mnemosynth
The Memory Synthesizer - An AI-powered voice & text conversational system that combines Whisper, Qwen (LLM), RAG (Retrieval Augmented Generation) and F5-TTS for personality simulation (in Spanish).

## Features

- **Audio Input**: Record and transcribe voice messages using Whisper
- **Chat Interface**: Interactive conversation with AI personality
- **RAG System**: Context-aware responses using PDF knowledge base
- **Voice Cloning**: Clone voices using reference audio samples
- **Text-to-Speech**: High-quality Spanish TTS using F5-TTS fine-tuned model
- **Real-time Processing**: Live audio generation and playback

### Docker Deployment

```bash
PENDING
```

### Assets Directory

Place your configuration files in `src/f5_tts/infer/Assets/`:

- **Initial_Prompt.txt**: System prompt for the AI
- **Personality.txt**: Personality description
- **Voice_Ref.wav**: Reference audio for voice cloning (WAV format)
- **Voice_Ref_Trans.txt**: Transcription of reference audio
- **/RAG/**: Directory containing PDF files for knowledge base

### Voice Reference

For best results with voice cloning:
- Use clear, high-quality audio (24kHz recommended)
- 10-15 seconds duration
- Single speaker
- Minimal background noise

### Models Used

- **TTS**: F5-TTS Spanish fine-tuned model by jpgallegoar
- **ASR**: OpenAI Whisper Large V3 Turbo (Spanish)
- **LLM**: Qwen 2.5 3B Instruct
- **Embeddings**: all-MiniLM-L6-v2 for RAG

### Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended (fallback to CPU/MPS)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 5GB+ for models and dependencies

### Performance Tips

- Use GPU for faster inference
- Adjust `nfe_step` parameter (16-32) for speed/quality trade-off
- Enable audio silence removal for cleaner output

# Credits
* [ΜΕΤΑΝΘΡΩΠΙΑ](https://github.com/METANTROP-IA) by this [Mnemosynth demo.](https://github.com/Metantrop-IA/Mnemosynth-03)
* [OpenAI](https://huggingface.co/openai) by Whisper.
* [Alibaba](https://huggingface.co/Qwen) by Qwen.                     
* [Yushen Chen](https://huggingface.co/SWivid) by the original [F5-TTS paper.](https://arxiv.org/abs/2410.06885)
* [jpgallegoar](https://github.com/jpgallegoar) by the Spanish fine-tuning of F5-TTS model.