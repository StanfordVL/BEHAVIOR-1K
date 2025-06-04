base_dir=/home/svl/Documents/Files/OG
task_name=picking_up_trash

task_dir="$base_dir/$task_name"
echo "Task directory: $task_dir"
# create raw directory if it doesn't exist
mkdir -p "$task_dir/raw"
cd $task_dir

# loop through all the files in the directory
for file in *; do
    if [[ $file == *.tar ]]; then
        # extract file name without extension
        dirname="${file%.tar}"
        # extract id from the file name which is the part after the last underscore
        id="${dirname##*_}"
        tar -xvf "$file" --directory "$task_dir"
        # remove the tar file
        rm "$task_dir/$file"
        cd "$dirname"
        # move .hdf5 files to the parent directory
        for hdf5_file in *.hdf5; do
            if [[ -f $hdf5_file ]]; then
                # get the file name without extension
                file_name="${hdf5_file%.*}"  
                # append the id to the file name
                mv "$hdf5_file" "$task_dir/raw/${file_name}_$id.hdf5"
            fi
        done
        # remove the directory 
        cd "$task_dir"
        rm -rf "$dirname"
    fi
done