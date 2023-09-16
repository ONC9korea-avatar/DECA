path_body=/workspace/DECA/TestSamples/examples_body/
path_head=/workspace/DECA/TestSamples/examples_head/
path_save=/workspace/DECA/output/

OLDPATH=$PATH
PATH=$PATH:"/usr/local/blender-2.79-linux-glibc219-x86_64/"

conda activate avatar
python third_parties/align-hair/main.py $path_save
conda deactivate

PATH=$PATH