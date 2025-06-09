#%%

import subprocess
import pandas as pd
import SimpleITK as sitk
import glob
import os
# import nibabel as nib
import numpy as np
# import nibabel as nib
# from totalsegmentator.python_api import totalsegmentator

def fix_nifti_direction(root):
    for nii_path in glob.glob(os.path.join(root, "*.nii.gz")):
        img = nib.load(nii_path)
        affine = img.affine.copy()
        # Set the rotation part of the affine to identity (I3)
        affine[:3, :3] = np.eye(3)
        fixed_img = nib.Nifti1Image(img.get_fdata(), affine)
        nib.save(fixed_img, nii_path)
        print(f"Fixed direction for {nii_path}")

def setOrthonormalDirection(root):
    for nii_path in glob.glob(os.path.join(root, "*.nii.gz")):
        nii = sitk.ReadImage(nii_path)
        if not nii.GetDirection() == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
            nii.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
            sitk.WriteImage(nii, nii_path)
            print(f"Updated direction for {nii_path}")

def totalSegmentation(rootPath: str,segmentationRoot:str,directImagePath: list = []):
    completedMasks = set()
    imageNames = sorted([f for f in os.listdir(rootPath) if f.endswith('.nii.gz')])[100:200]
    print("Image names:", imageNames[0])
    # imageNamesDf = pd.DataFrame(imageNames, columns=['Image'])
    # imageNamesDf.to_csv('imageNames.csv', index=False)
    # print("Image names:", len(imageNames))
    imagePaths = [imageName for imageName in directImagePath]
    # print("Image paths:", len(imagePaths))
    # if not os.path.exists(os.path.join(rootPath, "segmentations")):
    print(segmentationRoot, "seg root")
    for imageName in imageNames:
        # Check if the image file exists
        imagePath = os.path.join(rootPath, imageName)
        if imagePath in completedMasks:
            print(f"Image {imageName} already processed.")
            continue
        maskPath = os.path.join(segmentationRoot, imageName)
        print(f"Image path: {imagePath}")
        # print(f"Mask path: {maskPath}")
        print(f"Processing {maskPath}...")
        # Run TotalSegmentator
        totalsegmentator(
        input=imagePath,
        output=maskPath,
        roi_subset=["pancreas"],
        device = "cpu",
        verbose= True,
        nr_thr_saving= 6,
        )
        completedMasks.add(imagePath)
        print(f"Segmentation completed for {imageName}")
#100891 ekei epsase
def organize_files_by_folder(root_path):
    """
    Organizes files in the root_path by creating a folder for each file based on its name
    and moves the file into the folder, renaming it to 'pancreas.nii.gz'.
    """
    for fname in os.listdir(root_path):
        file_path = os.path.join(root_path, fname)
        if os.path.isfile(file_path):  # Check if it's a file
            # Create a folder based on the file name (without extension)
            folder_name = os.path.splitext(fname)[0]
            folder_path = os.path.join(root_path, folder_name)
            os.makedirs(folder_path, exist_ok=True)  # Ensure the folder exists

            # Move the file into the folder and rename it
            new_file_path = os.path.join(folder_path, "pancreas.nii.gz")
            os.rename(file_path, new_file_path+".gz")
            print(f"Moved {fname} to {new_file_path}")
def create_csv_for_pyradiomics(images_path, segs_path, output_csv_path):
    """
    Create a CSV file for pyradiomics --verbosity 2 from the given image and segmentation paths.
    """

    imageNames = sorted([f for f in os.listdir(images_path) if f.endswith('.nii.gz')])
    maskNames = sorted([f for f in os.listdir(segs_path) if f.endswith('.nii.gz')])
    print("Image names:", len(imageNames))
    print("Mask names:", len(maskNames))
    dataframe = pd.DataFrame(columns=['Image', 'Mask'])
    missingMasks = set(imageNames) - set(maskNames)
    print("Missing masks:", missingMasks)
    # missingDf = pd.DataFrame(columns=['Image'],data={'Image': list(missingMasks)})
    # missingDf.to_csv('missing_masks3.csv', index=False)
    for i in range(len(imageNames)):
        image_path = os.path.join(images_path, imageNames[i])
        mask_path = os.path.join(segs_path, maskNames[i])
        dataframe = pd.concat([dataframe,pd.DataFrame({'Image':image_path,'Mask':mask_path}, index=[0])], ignore_index=True)
        print(f"Image: {image_path}, Mask: {mask_path}")
    dataframe.to_csv(output_csv_path, index=False)


def zenodoRefactorStructure(root):
    """
    Refactor the structure of the dataset to match the Zenodo format.
    """
    i = 0
    for dirpath, dirnames, filenames in os.walk(root):
       
        for fileName in filenames:
            if fileName.endswith('.nii.gz'):
                # Get the full path of the file
                i= i + 1
                print(f"Processing file: {fileName} at {dirpath}")

                src = os.path.join(dirpath, fileName)
                dst= "/home/p/petridisp/pancaim/segbatch3/" + dirpath.split("/")[-1]
                print(f"Renaming {src} to {dst}")
                os.rename(
                    src,
                    dst
                )
    print(f"Total files in {dirpath}: {i}")


def convertMasksToInteger(segmentationRoot):
    """
    Convert segmentation masks to integer type.
    """
    for nii_path in glob.glob(os.path.join(segmentationRoot, "*.nii.gz")):
        img = nib.load(nii_path)
        data = img.get_fdata()
        data = data.astype(np.int8)  # Convert to int16
        new_img = nib.Nifti1Image(data, img.affine, img.header)
        nib.save(new_img, nii_path)
        print(f"Converted {nii_path} to int16")

def dispatchSingleExtraction(problematicDF,outputpath,paramsYaml):
    """
    Dispatch a single extraction for problematic cases.
    """
    df = problematicDF
    output_dir = outputpath
    params_yaml = paramsYaml

    for idx, row in df.iterrows():
        image = row['Image']
        mask = row['Mask']
        output_csv = f"{output_dir}"
        cmd = [
        "pyradiomics",
        "--verbosity", "5",
        image,
        mask,
        "-o", output_csv,
        "-f", "csv",
        "-p", params_yaml
    ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd)

def getEmptyRadiomicRows(csv_path, output_csv_path):
        """
        Get rows from the CSV file that have empty radiomic values.
        """
        df = pd.read_csv(csv_path)
        empty_rows = df[df.isnull().any(axis=1)]
        # empty_rows.to_csv(output_csv_path, index=False)
        print(f"Empty radiomic rows saved to {output_csv_path}")
if __name__ == "__main__":
    print("Starting batch mode processing...")
    print("--- DEBUG INFO (Python) ---")
    print(f"SLURM_JOB_ID: {os.getenv('SLURM_JOB_ID')}")
    print(f"SLURM_PROCID: {os.getenv('SLURM_PROCID')}")
    print(f"SLURM_ARRAY_TASK_ID: {os.getenv('SLURM_ARRAY_TASK_ID')}")
    print(f"SLURM_NTASKS: {os.getenv('SLURM_NTASKS')}")
    print(f"HOSTNAME: {os.getenv('HOSTNAME')}")
    print("--- END DEBUG INFO (Python) ---")

    #%%
    #------------FIND ERRONEOUS SEGMENTATIONS AND RETRY
    # import pandas as pd
    # getEmptyRadiomicRows("/home/p/petridisp/pancaim/batch2_200.csv", "/home/p/petridisp/pancaim/empty_radiomic_rows.csv")
    # df = pd.read_csv("/home/p/petridisp/pancaim/batch2_25.csv")
    # empty_rows = df[df.isnull().any(axis=1)]
    # dispatchSingleExtraction(empty_rows, "/home/p/petridisp/pancaim/batch2_25.csv", "/home/p/petridisp/radiomic/params2.yaml")
    #--------------END ERRONESOUS SEGMENTATIONS AND RETRY
    # create_csv_for_pyradiomics('/home/p/ppetrd/pancaim/batch1', '/home/p/ppetrd/pancaim/segmentations_batch1', 'home/p/ppetrd/pancaim/batch1/pyradiomics_input.csv')
    #     # Read the image and segmentation files
    #     image = sitk.ReadImage(ct_path)
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # print(current_dir)
    # imagePath = os.path.join("/home/p/petridisp", "pancaim", "batch2")
    # print(imagePath)
    # segPath = os.path.join("/home/p/petridisp", "pancaim", "segmentations_batch2")
    # print(segPath)
    # outputPath = os.path.join(current_dir, "pancaimBatch2.csv")
    # # fix_nifti_direction("/home/p/ppetrd/thesisCode/Image_Set1")
   
    # print("i read image")
    root = "/home/p/petridisp/pancaim/batch4"
    rootZenodoMasks = "/home/p/petridisp/pancaim/segmentations_batch3"
    # maskRoot = "/home/p/ppetrd/pancaim/segmentations_batch1"
    # create_csv_for_pyradiomics(root, maskRoot,"")
    # directImagePath = pd.read_csv( "missing_masks2.csv")['Image'].tolist()
    
    segmentationRoot = "/home/p/petridisp/pancaim/segmentations_batch4"
    # create_csv_for_pyradiomics(root, segmentationRoot,"")
    totalSegmentation(root,segmentationRoot=segmentationRoot)
    # organize_files_by_folder(segmentationRoot)

#---------------zenodo batch refactor----------------
    # zenodoRefactorStructure(rootZenodoMasks)
    # setOrthonormalDirection(root)
    # fix_nifti_direction(root)
    # print("Done processing batch mode.")
#---------------check segmentation labels----------------
#%%
fileName ="/home/p/petridisp/pancaim/segmentations_batch2/101106_00001_0000.nii.gz"
import nibabel as nib
import numpy as np
img = nib.load(fileName)
data = img.get_fdata()
print("Unique values:", np.unique(data))
print("Any label 1?", np.any(data == 1))

# #%%
# root = "/home/p/petridisp/pancaim/batch2"
# segmentationRoot = "/home/p/petridisp/pancaim/segmentations_batch2"
#create csv for pyradiomics
# create_csv_for_pyradiomics(root, segmentationRoot,"/home/p/petridisp/pancaim/batch2.csv")
# %%
import nibabel as nib

# convertMasksToInteger(segmentationRoot)
# %%


# %%
