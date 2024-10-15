import argparse
import numpy as np
import json
import pandas as pd

COMETQE_MODEL = "Unbabel/wmt22-cometkiwi-da"
COMETQE_BATCH_SIZE = 64
COMET_MODEL = "Unbabel/wmt22-comet-da"
COMET_BATCH_SIZE = 64

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hyp", type=str)
    parser.add_argument("datasetpath", type=str)
    parser.add_argument("--no-lexical-metrics", action="store_true")
    parser.add_argument("--no-comet", action="store_true")
    parser.add_argument("--no-comet-qe", action="store_true")
    parser.add_argument("--save-segment-level", default=None)
    parser.add_argument("--save-corpus-level", default=None)

    args = parser.parse_args()

    # load sources and references with pandas
    dataset_df = pd.read_csv(args.datasetpath, sep="\t")
    refs = dataset_df["targetString"].to_list()
    srcs = dataset_df["sourceString"].to_list()

    # load hypotheses from json
    with open(args.hyp, 'r') as hyp_f:
        hyps = json.load(hyp_f)

    # TODO: need to repeat the files dor the srcs/refs, if needed
    
    n_samples = int(len(hyps)/len(srcs))
    print('\nn_samples:', n_samples)
    refs = [entry for entry in refs for _ in range(n_samples)]
    srcs = [entry for entry in srcs for _ in range(n_samples)]
    print('\nRepated srcs and refs')

    sentence_metrics = [[] for _ in range(len(refs))]

    if not args.no_lexical_metrics:
        import sacrebleu

        # gets corpus-level non-ml evaluation metrics
        # corpus-level BLEU
        bleu = sacrebleu.metrics.BLEU()
        corpus_bleu = bleu.corpus_score(hyps, [refs])
        bleu_signature = str(bleu.get_signature())
        print(corpus_bleu)
        # corpus-level chrF
        chrf = sacrebleu.metrics.CHRF()
        corpus_chrf = chrf.corpus_score(hyps, [refs])
        chrf_signature = str(chrf.get_signature())
        print(corpus_chrf)

        if args.save_segment_level is not None:
            # gets sentence-level non-ml metrics
            for i, (hyp, ref) in enumerate(zip(hyps, refs)):
                sentence_metrics[i].append(
                    ("bleu", sacrebleu.sentence_bleu(hyp, [ref]).score)
                )
                sentence_metrics[i].append(
                    ("chrf", sacrebleu.sentence_chrf(hyp, [ref]).score)
                )
        
    if not args.no_comet_qe:
        from comet import download_model, load_from_checkpoint

        # download comet and load
        comet_path = download_model(COMETQE_MODEL)
        comet_model = load_from_checkpoint(comet_path)

        print("Running COMET evaluation...")
        comet_input = [
            {"src": src, "mt": mt} for src, mt in zip(srcs, hyps)
        ]
        # sentence-level and corpus-level COMET
        comet_output = comet_model.predict(
            comet_input, batch_size=COMETQE_BATCH_SIZE
        )

        comet_sentscores = comet_output.scores
        comet_score = comet_output.system_score

        for i, comet_sentscore in enumerate(comet_sentscores):
            sentence_metrics[i].append(("comet", comet_sentscore))

        corpus_cometkiwi = comet_score
        print(f"COMETKIWI = {comet_score:.4f}")

    if not args.no_comet:
        from comet import download_model, load_from_checkpoint

        # download comet and load
        comet_path = download_model(COMET_MODEL)
        comet_model = load_from_checkpoint(comet_path)

        print("Running COMET evaluation...")
        comet_input = [
            {"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(srcs, hyps, refs)
        ]
        # sentence-level and corpus-level COMET
        comet_output = comet_model.predict(
            comet_input, batch_size=COMET_BATCH_SIZE
        )

        comet_sentscores = comet_output.scores
        comet_score = comet_output.system_score

        for i, comet_sentscore in enumerate(comet_sentscores):
            sentence_metrics[i].append(("comet", comet_sentscore))

        corpus_comet = comet_score
        print(f"COMET = {comet_score:.4f}")

    # saves segment-level scores to the disk
    if args.save_segment_level is not None:
        with open(args.save_segment_level, "w") as f:
            if not args.no_comet_qe:
                print(" ".join(f"{metric_name}" for metric_name in ["bleu", "chrf", "cometkiwi", "comet"]), file=f)
            else:
                print(" ".join(f"{metric_name}" for metric_name in ["bleu", "chrf", "comet"]), file=f)

            for metrics in sentence_metrics:
                print(
                    " ".join(
                        f"{value}" for metric_name, value in metrics
                    ),
                    file=f,
                )

    # saves corpus-level scores to the disk
    if args.save_corpus_level is not None:
        with open(args.save_corpus_level, "w") as f:
            print("\n".join(f"{metric_value}" for metric_value in [corpus_bleu, corpus_chrf]), file=f)
            if not args.no_comet_qe:
                print(f"CometKiwi = {corpus_cometkiwi}", file=f)
            print(f"Comet = {corpus_comet}", file=f)
            print("non-ml metric signatures", file=f)
            print("\n".join(f"{metric_signature}" for metric_signature in [bleu_signature, chrf_signature]), file=f)

if __name__ == "__main__":
    main()