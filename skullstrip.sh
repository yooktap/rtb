#!/bin/sh
export PATH="/Users/yookta/ANTs/bin/:$PATH"
for type in DL SpD TD; do
    Output_dir="../procdata/${type}_stripped"
    mkdir -p $Output_dir
    Input_dir="../sampledata/$type"
     
    for infile in $Input_dir/*; do
        echo "$infile"
        outfile=`basename $infile`
        echo "antsBrainExtraction.sh -d 3 -a $infile -e ../Template/MNI152_T1_1mm-2.nii -m ../Template/MNI152_T1_1mm_Brain_Mask.nii -o $Output_dir/$outfile"
        antsBrainExtraction.sh -d 3 -a $infile -e ../Template/MNI152_T1_1mm-2.nii -m ../Template/MNI152_T1_1mm_Brain_Mask.nii -o $Output_dir/$outfile
    done
done
