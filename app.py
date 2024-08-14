from argparse import Namespace
from src.watermark_demo import main

def get_default_args():
    """Returns the default arguments as a Namespace object."""
    default_arg_dict = {
        'run_gradio': True,
        'run_extended': False,
        'demo_public': False,
        'model_name_or_path': 'bigscience/bloom-560m', #'google/gemma-2-2b-it', #'meta-llama/Meta-Llama-3-8B',
        'load_fp16': False,
        'prompt_max_length': None,
        'max_new_tokens': 200,
        'generation_seed': 123,
        'use_sampling': True,
        'n_beams': 1,
        'sampling_temp': 0.7,
        'use_gpu': False,
        'seeding_scheme': 'simple_1',
        'gamma': 0.25,
        'delta': 2.0,
        'normalizers': '',
        'skip_repeated_bigrams': False,
        'ignore_repeated_ngrams': False,
        'detection_z_threshold': 4.0,
        'select_green_tokens': True,
        'skip_model_load': True,
        'seed_separately': True,
    }
    
    args = Namespace()
    args.__dict__.update(default_arg_dict)
    return args

if __name__ == "__main__":
    args = get_default_args()
    main(args)
