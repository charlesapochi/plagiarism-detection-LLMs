import sys
from functools import partial
import gradio as gr
import torch
from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList)
from algorithm.watermark_engine import LogitsProcessorWithWatermark, WatermarkAnalyzer
from algorithm.extended_watermark_engine import LogitsProcessorWithWatermarkExtended, WatermarkAnalyzerExtended
from components.utils import process_args, get_default_prompt, display_prompt, display_results, parse_args, list_format_scores, get_default_args


def run_gradio(args, model=None, device=None, tokenizer=None):
    """Define and launch with gradio"""
    generate_partial = partial(generate, model=model, device=device, tokenizer=tokenizer)
    detect_partial = partial(analyze, device=device, tokenizer=tokenizer)

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), css="footer{display:none !important}") as demo:
        with gr.Row():
            with gr.Column(scale=9):
                gr.Markdown(
                    """
                    ## Plagiarism detection for Large Language Models through watermarking
                    """
                    )
            with gr.Column(scale=2):
                algorithm = gr.Radio(label="Watermark Algorithm", info="which algorithm would you like to use?", choices=["basic", "advance"], value=("advance" if args.run_extended else "basic"))    
        
        gr.Markdown(f"Language model: {args.model_name_or_path} {'(float16 mode)' if args.load_fp16 else ''}")
        
        default_prompt = args.__dict__.pop("default_prompt")
        session_args = gr.State(value=args)

        with gr.Tab("Generate and Detect"):
            
            with gr.Row():
                prompt = gr.Textbox(label=f"Prompt", interactive=True,lines=10,max_lines=10, value=default_prompt)
            with gr.Row():
                generate_btn = gr.Button("Generate")
            with gr.Row():
                with gr.Column(scale=2):
                    output_without_watermark = gr.Textbox(label="Output Without Watermark", interactive=False,lines=14,max_lines=14)
                with gr.Column(scale=1):
                    without_watermark_detection_result = gr.Dataframe(headers=["Metric", "Value"], interactive=False,row_count=7,col_count=2)
            with gr.Row():
                with gr.Column(scale=2):
                    output_with_watermark = gr.Textbox(label="Output With Watermark", interactive=False,lines=14,max_lines=14)
                with gr.Column(scale=1):
                    with_watermark_detection_result = gr.Dataframe(headers=["Metric", "Value"],interactive=False,row_count=7,col_count=2)

            redecoded_input = gr.Textbox(visible=False)
            truncation_warning = gr.Number(visible=False)
            def truncate_prompt(redecoded_input, truncation_warning, orig_prompt, args):
                if truncation_warning:
                    return redecoded_input + f"\n\n[Prompt was truncated before generation due to length...]", args
                else: 
                    return orig_prompt, args
        
        with gr.Tab("Detector Only"):
            with gr.Row():
                with gr.Column(scale=2):
                    detection_input = gr.Textbox(label="Text to Analyze", interactive=True,lines=14,max_lines=14)
                with gr.Column(scale=1):
                    detection_result = gr.Dataframe(headers=["Metric", "Value"], interactive=False,row_count=7,col_count=2)
            with gr.Row():
                    detect_btn = gr.Button("Detect")
        
        gr.HTML("""
                <p style="color: gray;">Built with 🤍 by Charles Apochi
                <br/>
                <a href="mailto:charlesapochi@gmail.com" style="text-decoration: none; color: orange;">Reach out</a>
                <p/>
                """)
        
        generate_btn.click(fn=generate_partial, inputs=[prompt,session_args], outputs=[redecoded_input, truncation_warning, output_without_watermark, output_with_watermark,session_args])
        redecoded_input.change(fn=truncate_prompt, inputs=[redecoded_input,truncation_warning,prompt,session_args], outputs=[prompt,session_args])
        output_without_watermark.change(fn=detect_partial, inputs=[output_without_watermark,session_args], outputs=[without_watermark_detection_result,session_args])
        output_with_watermark.change(fn=detect_partial, inputs=[output_with_watermark,session_args], outputs=[with_watermark_detection_result,session_args])
        detect_btn.click(fn=detect_partial, inputs=[detection_input,session_args], outputs=[detection_result, session_args])

        # State management logic
        def update_algorithm(session_state, value):
            if value == "advance":
                session_state.run_extended = True
                # args.run_extended = True
            elif value == "basic":
                session_state.run_extended = False
                # args.run_extended = False
            return session_state

        
        algorithm.change(update_algorithm,inputs=[session_args, algorithm], outputs=[session_args])
        
    demo.launch(share=args.demo_public)


def load_model(args):
    """Load and return the model and tokenizer"""
    args.is_decoder_only_model = True
    
    model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                # device_map="auto",
                # torch_dtype=torch.float16,
            )

    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.load_fp16: 
            pass
        else: 
            model = model.to(device)
    else:
        device = "cpu" #"mps" if args.run_extended else "cpu"
    
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    return model, tokenizer, device

def generate(prompt, args, model=None, device=None, tokenizer=None):
    print(f"Generating with {args}")
    
    if args.run_extended: 
        watermark_processor = LogitsProcessorWithWatermarkExtended(vocab=list(tokenizer.get_vocab().values()),
                                                    gamma=args.gamma,
                                                    delta=args.delta,
                                                    seeding_scheme=args.seeding_scheme,
                                                    select_green_tokens=args.select_green_tokens)
    else:
        watermark_processor = LogitsProcessorWithWatermark(vocab=list(tokenizer.get_vocab().values()),
                                                    gamma=args.gamma,
                                                    delta=args.delta,
                                                    seeding_scheme=args.seeding_scheme,
                                                    select_green_tokens=args.select_green_tokens)
    
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)

    if args.use_sampling:
        gen_kwargs.update(dict(
            do_sample=True, 
            top_k=0,
            temperature=args.sampling_temp,
        ))
    else:
        gen_kwargs.update(dict(
            num_beams=args.n_beams,
        ))

    generate_without_watermark = partial(
        model.generate,
        **gen_kwargs
    )
    generate_with_watermark = partial(
        model.generate,
        logits_processor=LogitsProcessorList([watermark_processor]), 
        **gen_kwargs
    )
    if args.prompt_max_length:
        pass
    elif hasattr(model.config,"max_position_embedding"):
        args.prompt_max_length = model.config.max_position_embeddings-args.max_new_tokens
    else:
        args.prompt_max_length = 2048-args.max_new_tokens
    
    tokd_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=args.prompt_max_length).to(device)
    truncation_warning = True if tokd_input["input_ids"].shape[-1] == args.prompt_max_length else False
    redecoded_input = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]

    torch.manual_seed(args.generation_seed)
    output_without_watermark = generate_without_watermark(**tokd_input)

    if args.seed_separately: 
        torch.manual_seed(args.generation_seed)
    output_with_watermark = generate_with_watermark(**tokd_input)

    if args.is_decoder_only_model:
        # need to isolate the newly generated tokens
        output_without_watermark = output_without_watermark[:,tokd_input["input_ids"].shape[-1]:]
        output_with_watermark = output_with_watermark[:,tokd_input["input_ids"].shape[-1]:]

    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]

    return (redecoded_input,
            int(truncation_warning),
            decoded_output_without_watermark, 
            decoded_output_with_watermark,
            args) 
            # decoded_output_with_watermark)

def analyze(input_text, args, device=None, tokenizer=None):
    
    detector_args = {
        "vocab": list(tokenizer.get_vocab().values()),
        "gamma": args.gamma,
        "delta": args.delta,
        "seeding_scheme": args.seeding_scheme,
        "select_green_tokens": args.select_green_tokens,
        "device": device,
        "tokenizer": tokenizer,
        "z_threshold": args.detection_z_threshold,
        "normalizers": args.normalizers,
    }
    if args.run_extended:
        detector_args["ignore_repeated_ngrams"] = args.ignore_repeated_ngrams
    else:
        detector_args["skip_repeated_bigrams"] = args.skip_repeated_bigrams

    if args.run_extended: 
        watermark_detector = WatermarkAnalyzerExtended(**detector_args)
    else: 
        watermark_detector = WatermarkAnalyzer(**detector_args)
        
    if args.run_extended:
        score_dict = watermark_detector.analyze(input_text)
        output = list_format_scores(score_dict, watermark_detector.z_threshold)
    else:
        if len(input_text)-1 > watermark_detector.min_prefix_len:
            score_dict = watermark_detector.analyze(input_text)
            # output = str_format_scores(score_dict, watermark_detector.z_threshold)
            output = list_format_scores(score_dict, watermark_detector.z_threshold)
        else:
            # output = (f"Error: string not long enough to compute watermark presence.")
            output = [["Error","string too short to compute metrics"]]
            output += [["",""] for _ in range(6)]
    
    return output, args

if __name__ == "__main__":
    args = parse_args()
    # args = get_default_args()
    # args = process_args(args)
    input_text = get_default_prompt()
    args.default_prompt = input_text

    if not args.skip_model_load:
        model, tokenizer, device = load_model(args)
    else:
        model, tokenizer, device = None, None, None

    if not args.skip_model_load:
        display_prompt(input_text)

        _, _, decoded_output_without_watermark, decoded_output_with_watermark, _ = generate(
            input_text, args, model=model, device=device, tokenizer=tokenizer
        )

        without_watermark_detection_result = analyze(
            decoded_output_without_watermark, args, device=device, tokenizer=tokenizer
        )

        with_watermark_detection_result = analyze(
            decoded_output_with_watermark, args, device=device, tokenizer=tokenizer
        )

        display_results(decoded_output_without_watermark, without_watermark_detection_result, args, with_watermark=False)
        display_results(decoded_output_with_watermark, with_watermark_detection_result, args, with_watermark=True)

    if args.run_gradio:
        run_gradio(args, model=model, tokenizer=tokenizer, device=device)
