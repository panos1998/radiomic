import pandas as pd
import SimpleITK as sitk
import glob
import os
import nibabel as nib
import numpy as np
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator

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
    imageNames = sorted([f for f in os.listdir(rootPath) if f.endswith('.nii.gz')])
    # imagePaths = [os.path.join(rootPath, imageName) for imageName in directImagePath]
    # if not os.path.exists(os.path.join(rootPath, "segmentations")):
    for imageName in directImagePath:
        imagePath = os.path.join(rootPath, imageName)
        maskPath = os.path.join(segmentationRoot, imageName)
        print(f"Image path: {imagePath}")
        print(f"Mask path: {maskPath}")
        
        # Run TotalSegmentator
        totalsegmentator(
        input=imagePath,
        output=maskPath,
        roi_subset=["pancreas"],
        )
        print(f"Segmentation completed for {imageName}")

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
    maskNames = sorted([f for f in os.listdir(segs_path) if f.endswith('.nii.gz') or f.endswith('.nii')])
    print("Image names:", len(imageNames))
    print("Mask names:", len(maskNames))
    dataframe = pd.DataFrame(columns=['Image', 'Mask'])
    missingMasks = set(imageNames) - set(maskNames)
    print("Missing masks:", missingMasks)
    missingDf = pd.DataFrame(columns=['Image'],data={'Image': list(missingMasks)})
    missingDf.to_csv('missing_masks.csv', index=False)
    # for i in range(len(imageNames)):
    #     image_path = os.path.join(images_path, imageNames[i])
    #     mask_path = os.path.join(segs_path, maskNames[i])
    #     dataframe = pd.concat([dataframe,pd.DataFrame({'Image':image_path,'Mask':mask_path}, index=[0])], ignore_index=True)
        # print(f"Image: {image_path}, Mask: {mask_path}")
    # dataframe.to_csv(output_csv_path, index=False)
if __name__ == "__main__":
    print("Starting batch mode processing...")
    # create_csv_for_pyradiomics('/home/p/ppetrd/pancaim/batch1', '/home/p/ppetrd/pancaim/segmentations_batch1', 'home/p/ppetrd/pancaim/batch1/pyradiomics_input.csv')
    #     # Read the image and segmentation files
    #     image = sitk.ReadImage(ct_path)
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # print(current_dir)
    # imagePath = os.path.join("/home/p/ppetrd", "Images_segs_Set2", "Image_Set2")
    # print(imagePath)
    # segPath = os.path.join("/home/p/ppetrd", "Images_segs_Set2", "Segmentation_Set2")
    # print(segPath)
    # outputPath = os.path.join(current_dir, "maskSegMatchingMUKERJE_SET2.csv")
    # # fix_nifti_direction("/home/p/ppetrd/thesisCode/Image_Set1")
    # create_csv_for_pyradiomics(imagePath, segPath, outputPath)
    # print("i read image")
    root = "/home/p/ppetrd/pancaim/batch1"
    maskRoot = "/home/p/ppetrd/pancaim/segmentations_batch1"
    # create_csv_for_pyradiomics(root, maskRoot,"")
    directImagePath = pd.read_csv( "missing_masks.csv")['Image'].tolist()
    segmentationRoot = "/home/p/ppetrd/pancaim/segmentations_batch1"
    # totalSegmentation(root,segmentationRoot=segmentationRoot, directImagePath=directImagePath)
    organize_files_by_folder(segmentationRoot)
    
