mkdir -p eval/data/meshes
cd eval/data/meshes
wget -O data.zip https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_models.zip
unzip data.zip
rm -rf data.zip models_eval models_fine
mv models ycbv
cd ycbv
num_meshes=$(ls | grep -c ".ply")
for i in `seq 1 $num_meshes`
do
    echo $i
    mesh_name=$(printf "obj_%06d.ply" $i)
    texture_name=$(printf "obj_%06d.png" $i)
    folder_name=$(printf "%d" $i)
    mkdir $folder_name
    mv $mesh_name $folder_name
    mv $texture_name $folder_name
done
rm -rf models_info.json
echo "Done"
