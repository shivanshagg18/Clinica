#!/bin/bash

#The MNI_template.nii.gz is NIFTI image 

#MNI_template = MNI152_T1_2mm_brain.nii.gz

#We will use a "for" loop to loop/iterate over each subject. ]

for subj in ./CAPS/*/*/*/*/*/*/*/*.nii.gz

    do
        echo 'Working on ' ${subj}
        echo  ${subj:0:-7}
        flirt -in ${subj}  -ref MNI152_T1_2mm_brain.nii.gz  -out  ${subj:0:-7}_Registered.nii.gz
    done