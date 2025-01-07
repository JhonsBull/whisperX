import os
import gc
import warnings
import numpy as np
import torch
from argparse import ArgumentParser
from typing import Dict, Any
from faster_whisper import TranscriptionResult, AlignedTranscriptionResult, load_model, load_align_model, align
from diarization_pipeline import DiarizationPipeline
from utils import LANGUAGES, TO_LANGUAGE_CODE, load_audio, get_writer, assign_word_speakers

def cli():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for transcriptions")
    parser.add_argument("--model_dir", type=str, default="./models", help="Directory to store downloaded models")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save output files")
    parser.add_argument("--output_format", type=str, default="txt", choices=["txt", "json", "srt", "vtt"], help="Output format")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for inference")
    parser.add_argument("--device_index", type=int, default=0, help="Index of the GPU to use (if applicable)")
    parser.add_argument("--compute_type", type=str, default="float32", choices=["float16", "float32", "int8"], help="Compute type for inference")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--align_model", type=str, default="large", help="Alignment model name")
    parser.add_argument("--interpolate_method", type=str, default="linear", choices=["linear", "cubic"], help="Interpolation method")
    parser.add_argument("--no_align", action="store_true", help="Disable alignment step")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="Task to perform")
    parser.add_argument("--return_char_alignments", action="store_true", help="Return character-level alignments")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token for diarization model")
    parser.add_argument("--vad_onset", type=float, default=0.5, help="VAD onset threshold")
    parser.add_argument("--vad_offset", type=float, default=0.5, help="VAD offset threshold")
    parser.add_argument("--chunk_size", type=int, default=30, help="Audio chunk size in seconds")
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization")
    parser.add_argument("--min_speakers", type=int, default=1, help="Minimum number of speakers for diarization")
    parser.add_argument("--max_speakers", type=int, default=5, help="Maximum number of speakers for diarization")
    parser.add_argument("--print_progress", action="store_true", help="Print progress during transcription")
    parser.add_argument("--threads", type=int, default=0, help="Number of threads for computation")
    parser.add_argument("--language", type=str, help="Language of the audio")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--temperature_increment_on_fallback", type=float, help="Increment temperature on fallback")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size for decoding")
    parser.add_argument("--patience", type=float, help="Beam search patience")
    parser.add_argument("--length_penalty", type=float, help="Beam search length penalty")
    parser.add_argument("--compression_ratio_threshold", type=float, default=0.5, help="Compression ratio threshold")
    parser.add_argument("--logprob_threshold", type=float, help="Log probability threshold")
    parser.add_argument("--no_speech_threshold", type=float, help="No speech probability threshold")
    parser.add_argument("--initial_prompt", type=str, help="Initial prompt for decoding")
    parser.add_argument("--suppress_tokens", type=str, default="-1", help="Comma-separated list of token IDs to suppress")
    parser.add_argument("--suppress_numerals", action="store_true", help="Suppress numeral tokens in decoding")
    parser.add_argument("--highlight_words", action="store_true", help="Highlight words in output")
    parser.add_argument("--max_line_count", type=int, help="Maximum number of lines per segment")
    parser.add_argument("--max_line_width", type=int, help="Maximum width of each line")
    parser.add_argument("audio", nargs="+", help="Audio file(s) to process")

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    batch_size: int = args.pop("batch_size")
    model_dir: str = args.pop("model_dir")
    output_dir: str = args.pop("output_dir")
    output_format: str = args.pop("output_format")
    device: str = args.pop("device")
    device_index: int = args.pop("device_index")
    compute_type: str = args.pop("compute_type")
    verbose: bool = args.pop("verbose")
    os.makedirs(output_dir, exist_ok=True)
    align_model: str = args.pop("align_model")
    interpolate_method: str = args.pop("interpolate_method")
    no_align: bool = args.pop("no_align")
    task: str = args.pop("task")
    if task == "translate":
        no_align = True
    return_char_alignments: bool = args.pop("return_char_alignments")
    hf_token: str = args.pop("hf_token")
    vad_onset: float = args.pop("vad_onset")
    vad_offset: float = args.pop("vad_offset")
    chunk_size: int = args.pop("chunk_size")
    diarize: bool = args.pop("diarize")
    min_speakers: int = args.pop("min_speakers")
    max_speakers: int = args.pop("max_speakers")
    print_progress: bool = args.pop("print_progress")

    if args["language"] is not None:
        args["language"] = args["language"].lower()
        if args["language"] not in LANGUAGES:
            if args["language"] in TO_LANGUAGE_CODE:
                args["language"] = TO_LANGUAGE_CODE[args["language"]]
            else:
                raise ValueError(f"Unsupported language: {args['language']}")

    if model_name.endswith(".en") and args["language"] != "en":
        if args["language"] is not None:
            warnings.warn(
                f"{model_name} is an English-only model but received '{args['language']}'; using English instead."
            )
        args["language"] = "en"
    align_language = args["language"] if args["language"] is not None else "en"

    temperature = args.pop("temperature")
    if (increment := args.pop("temperature_increment_on_fallback")) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]

    faster_whisper_threads = 4
    if (threads := args.pop("threads")) > 0:
        torch.set_num_threads(threads)
        faster_whisper_threads = threads

    asr_options = {
        "beam_size": args.pop("beam_size"),
        "patience": args.pop("patience"),
        "length_penalty": args.pop("length_penalty"),
        "temperatures": temperature,
        "compression_ratio_threshold": args.pop("compression_ratio_threshold"),
        "log_prob_threshold": args.pop("logprob_threshold"),
        "no_speech_threshold": args.pop("no_speech_threshold"),
        "condition_on_previous_text": False,
        "initial_prompt": args.pop("initial_prompt"),
        "suppress_tokens": [int(x) for x in args.pop("suppress_tokens").split(",")],
        "suppress_numerals": args.pop("suppress_numerals"),
    }

    writer = get_writer(output_format, output_dir)
    word_options = ["highlight_words", "max_line_count", "max_line_width"]
    if no_align:
        for option in word_options:
            if args[option]:
                parser.error(f"--{option} not possible with --no_align")
    if args["max_line_count"] and not args["max_line_width"]:
        warnings.warn("--max_line_count has no effect without --max_line_width")
    writer_args = {arg: args.pop(arg) for arg in word_options}
    results = []
    tmp_results = []
    model = load_model(
        model_name,
        device=device,
        device_index=device_index,
        download_root=model_dir,
        compute_type=compute_type,
        language=args["language"],
        asr_options=asr_options,
        vad_options={"vad_onset": vad_onset, "vad_offset": vad_offset},
        task=task,
        threads=faster_whisper_threads,
    )

    for audio_path in args.pop("audio"):
        audio = load_audio(audio_path)
        print(">>Performing transcription...")
        result: TranscriptionResult = model.transcribe(
            audio,
            batch_size=batch_size,
            chunk_size=chunk_size,
            print_progress=print_progress,
            verbose=verbose,
        )
        results.append((result, audio_path))

    del model
    gc.collect()
    torch.cuda.empty_cache()

    if not no_align:
        tmp_results = results
        results = []
        align_model, align_metadata = load_align_model(align_language, device, model_name=align_model)
        for result, audio_path in tmp_results:
            input_audio = audio_path if len(tmp_results) > 1 else audio
            if align_model is not None and len(result["segments"]) > 0:
                if result.get("language", "en") != align_metadata["language"]:
                    print(f"New language found ({result['language']})! Loading new alignment model...")
                    align_model, align_metadata = load_align_model(result["language"], device)
                print(">>Performing alignment...")
                result: AlignedTranscriptionResult = align(
                    result["segments"],
                    align_model,
                    align_metadata,
                    input_audio,
                    device,
                    interpolate_method=interpolate_method,
                    return_char_alignments=return_char_alignments,
                    print_progress=print_progress,
                )
            results.append((result, audio_path))

        del align_model
        gc.collect()
        torch.cuda.empty_cache()

    if diarize:
        if hf_token is None:
            print("Warning: No --hf_token provided for diarization model.")
        tmp_results = results
        print(">>Performing diarization...")
        results = []
        diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)
        for result, input_audio_path in tmp_results:
            diarize_segments = diarize_model(input_audio_path, min_speakers=min_speakers, max_speakers=max_speakers)
            result = assign_word_speakers(diarize_segments, result)
            results.append((result, input_audio_path))

    for result, audio_path in results:
        result["language"] = align_language
        writer(result, audio_path, writer_args)

if __name__ == "__main__":
    cli()
