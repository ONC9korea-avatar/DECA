data_path=/workspace/DECA/TestSamples/examples_head
save_path=/workspace/DECA/output
skin_path=/workspace/DECA/TestSamples/examples_skin_only

conda activate scikit-learn
python third_parties/texture-synthesis/main.py $data_path $save_path $skin_path
conda deactivate