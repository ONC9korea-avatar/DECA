face_parsing_path=/workspace/DECA/third_parties/face-parsing.PyTorch
data_path=/workspace/DECA/TestSamples/examples

conda activate face_alignment
cd $face_parsing_path
python test.py $data_path
cd - > /dev/null
conda deactivate