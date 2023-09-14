data_path=/workspace/DECA/TestSamples/examples_head
save_path=/workspace/DECA/output

conda activate deca-38
python third_parties/texture-synthesis/main.py $data_path $save_path
conda deactivate