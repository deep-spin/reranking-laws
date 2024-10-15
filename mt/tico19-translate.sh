repo_path=
env_path=
test_set='tico19'
temperature=1.0
top_p=1.0
n_samples=50
src_lang='en'

source $env_path/bin/activate
for tgt_lang in 'pt-BR' 'es-LA' 'ru'
do
    for split in 'dev' 'test'
    do
    output_file_dir=$repo_path/results-$test_set/$split/$src_lang-$tgt_lang
    mkdir $repo_path/results-$test_set/
    mkdir $repo_path/results-$test_set/$split
    mkdir $output_file_dir
    output_file=$output_file_dir/towerinstruct13b-temp$temperature-topp$top_p-n$n_samples

    python3 $repo_path/generate_with_vllm.py -o $output_file \
        --num_return_sequences $n_samples \
        --temperature $temperature \
        --top_p $top_p \
        --src_lang $src_lang \
        --tgt_lang $tgt_lang \
        --test-set $test_set \
        --split $split
        
    done
done
