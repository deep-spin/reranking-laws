from vllm import LLM, SamplingParams
import pandas as pd
import numpy as np
import argparse
import json

LANG_TABLE = {
    "en": "English",
    "ru": "Russian",
    "pt-BR": "Portuguese",
    "es-LA": "Spanish",
}

def get_data(src_lang, tgt_lang, test_set="tico19", split="dev"):
    if test_set == "tico19":
        df = pd.read_csv(f"/mnt/data-poseidon/antoniofarinhas/data/tico19-testset/{split}/{split}.{src_lang}-{tgt_lang}.tsv", sep="\t")
        src_text =  df["sourceString"].to_list()
        tgt_text = df["targetString"].to_list()
    return src_text, tgt_text

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Unbabel/TowerInstruct-13B-v0.2")
    parser.add_argument("-o", "--output_path", required=True, type=str)
    parser.add_argument("--max_new_tokens", default=128, type=int)
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument("--num_return_sequences", default=1, type=int)
    parser.add_argument("--top_k", default=-1, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--prompt_choice", default="tower", type=str) 
    parser.add_argument("--src_lang", required=True, type=str) 
    parser.add_argument("--tgt_lang", required=True, type=str) 
    parser.add_argument("--test-set", default="tico19", type=str)
    parser.add_argument("--split", default="dev", type=str)


    args = parser.parse_args()
    return args

def main(args):
    model_path = args.model_path
    output_path = args.output_path
    test_set = args.test_set
    split = args.split
    
    # get data
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    src_text, tgt_text = get_data(src_lang, tgt_lang, test_set, split)

    # llm related stuff
    llm = LLM(model=f"{model_path}", trust_remote_code=True)
    sampling_params = SamplingParams(temperature=args.temperature, 
                                     max_tokens=args.max_new_tokens,
                                     top_p=args.top_p,
                                     top_k=args.top_k,
                                     use_beam_search=(args.num_beams > 1))
    
    if args.prompt_choice == "tower":
        prompts =  [f'<|im_start|>user\nTranslate the following {LANG_TABLE[src_lang]} source text to {LANG_TABLE[tgt_lang]}:\n{LANG_TABLE[src_lang]}: {x}\n{LANG_TABLE[tgt_lang]}: <|im_end|>\n<|im_start|>assistant\n' for x in src_text]
    else:
        pass
    
    # more than one sample
    repeated_prompts = []
    for prompt in prompts:
        repeated_prompts += [prompt] * args.num_return_sequences

    outputs_raw = llm.generate(repeated_prompts, sampling_params)
    output = [sample.text for x in outputs_raw for sample in x.outputs]

    print('len(outputs_raw), len(output): ', len(outputs_raw), len(output))

    # save as txt file
    output_file = open(f"{output_path}", "w", encoding="utf-8")
    for line in output:
        output_file.write(line.replace("\n","\\n"))
        output_file.write('\n')

    # save as json
    with open(f"{output_path}.json", 'w') as fout:
        json.dump(output, fout)


if __name__ == "__main__":
    args = get_args()
    main(args)