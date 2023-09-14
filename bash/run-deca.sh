data_path=/workspace/DECA/TestSamples/examples_masked
save_path=/workspace/DECA/TestSamples/examples_head

copy_path=$data_path/../exsamples_mesh

conda activate deca-38
python demos/demo_reconstruct.py -i $data_path -s $save_path --saveObj True --saveVis False --rasterizer_type pytorch3d
conda deactivate