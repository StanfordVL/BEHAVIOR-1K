base_dir=/home/svl/Documents/Files/behavior
download_dir=/home/svl/Downloads
task_name=can_meat

task_dir="${base_dir}/raw/${task_name}"
echo "Task directory: $task_dir"
# create raw directory if it doesn't exist
mkdir -p $task_dir
cd $download_dir

# Initialize counter for unique IDs
instance_id=0

# loop through all the files in the directory
for file in *; do
    if [[ $file == *.tar ]]; then
        # extract file name without extension
        dirname="${file%.tar}"
        tar -xvf $file --directory $task_dir
        # remove the tar file
        rm $file
        cd $task_dir/$dirname
        # move .hdf5 files to the task directory
        for hdf5_file in *.hdf5; do
            if [[ -f $hdf5_file ]]; then
                # get the file name without extension
                file_name="${hdf5_file%.*}"  
                # append the id to the file name
                # Make instance_id 3 digits long
                instance_id_str=$(printf "%03d" $instance_id)
                mv $hdf5_file "${task_dir}/${file_name}_${instance_id_str}.hdf5"
                instance_id=$((instance_id + 1))
            fi
        done
        # remove the directory 
        cd $download_dir
        rm -rf $task_dir/$dirname
    fi
done