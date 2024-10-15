repo_path=
env_path=
env_comet_mbr_path=
dataset_path=
test_set='tico19'
temperature=1.0
top_p=1.0
n_samples=50
src_lang='en'


for tgt_lang in  'pt-BR' 'es-LA' 'ru'
do
    for split in 'test' 'dev'
    do

    dataset_file=$dataset_path/$split/$split.$src_lang-$tgt_lang.tsv
    output_file_dir=$repo_path/results-$test_set/$split/$src_lang-$tgt_lang
    output_file=$output_file_dir/towerinstruct13b-temp$temperature-topp$top_p-n$n_samples
    
    metrics_dir=$repo_path/results-$test_set/$split/$src_lang-$tgt_lang/metrics
    metrics_corpus=$metrics_dir/corpus/towerinstruct13b-temp$temperature-topp$top_p-n$n_samples
    metrics_segment=$metrics_dir/segment/towerinstruct13b-temp$temperature-topp$top_p-n$n_samples
    mkdir $metrics_dir
    mkdir $metrics_dir/corpus
    mkdir $metrics_dir/segment

    source $env_path/bin/activate
    python3 $repo_path/score.py \
        $output_file.json \
        $dataset_file \
        --save-segment-level $metrics_segment \
        --save-corpus-level $metrics_corpus \

    if [ $n_samples != 1 ]; then
        
        source_file=$repo_path/results-$test_set/$split/$src_lang-$tgt_lang/src.txt
        target_file=$repo_path/results-$test_set/$split/$src_lang-$tgt_lang/ref.txt
        python3 $repo_path/get_src_and_ref.py \
            $dataset_file \
            $source_file \
            $target_file

        translations_mbr=$output_file-mbr
        metrics_corpus_mbr=$metrics_corpus-mbr
        metrics_segment_mbr=$metrics_segment-mbr

        source $env_comet_mbr_path/bin/activate
        comet-mbr -s $source_file \
            -t $output_file \
            --num_sample $n_samples \
            -o $translations_mbr

    else
        echo "Single translation: not doing MBR" 
    fi

    done
done