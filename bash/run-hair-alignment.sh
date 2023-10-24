path_hair_obj='/workspace/DECA/third_parties/align-hair/hair_mm.obj'
path_hair_mask='/workspace/DECA/TestSamples/examples_hair_only'
path_save=/workspace/DECA/output/

OLDPATH=$PATH
PATH=$PATH:"/usr/local/blender-2.79-linux-glibc219-x86_64/"

conda activate avatar
python third_parties/align-hair/main.py $path_save $path_hair_obj $path_hair_mask
conda deactivate

PATH=$PATH