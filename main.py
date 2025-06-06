#%%
import os
import re
import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
import pydicom as pdcm
from radiomics import featureextractor
import os

import SimpleITK as sitk
import six
from experimentClass import Experiment
import nrrd
from radiomics import featureextractor
#TODO: na meleitiso to simpleITK spacing. voxel size,resampling na dw ti paizei

def read_image_files(input_folder):
    """
    Reads all image files in the input folder and returns them as a list.
    """
    image_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
                image_files.append(os.path.join(root, file))
    return image_files
def process_and_save_images(experiment, image, segmentation):
    """
    Reads, resamples, and saves all images and segmentations using the given Experiment instance.
    """
    # Resample the image and segmentation to isotropic spacing
    resampledImage = experiment.isotropicResampleImage(image)
    resampledSegmentation = experiment.isotropicResampleSegmentation(segmentation)
    print("i resampled both")
    # Hounsfield windowing (uncomment if needed)
    # image = experiment.hounsfieldWindowing(image)
    print("i windowed ct")
    # Save the resampled image and segmentation
    # experiment.storeImage(image, os.path.join(output_folder_ct, f"resampled_image_{i}.nii.gz"))
    # experiment.storeImage(segmentation, os.path.join(output_folder_seg, f"resampled_segmentation_{i}.nii.gz"))
    print("Image min/max:", sitk.GetArrayFromImage(resampledImage).min(), sitk.GetArrayFromImage(resampledImage).max())
    print("Mask unique values:", np.unique(sitk.GetArrayFromImage(resampledSegmentation)))
    print("Mask sum:", sitk.GetArrayFromImage(resampledSegmentation).sum())
    print("Image size:", resampledImage.GetSize(), "Mask size:", resampledSegmentation.GetSize())
    print("Image spacing:", resampledImage.GetSpacing(), "Mask spacing:", resampledSegmentation.GetSpacing())
    print(type(resampledImage), type(resampledSegmentation))
    print(resampledImage.GetSize(), resampledSegmentation.GetSize())
    print(resampledImage.GetSpacing(), resampledSegmentation.GetSpacing())
    print(resampledSegmentation.GetPixelIDTypeAsString())
    return resampledImage, resampledSegmentation
      
# Main script
 #%%
if __name__ == "__main__":
    try:
        os.makedirs("numpyFmaps", exist_ok=True)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        input_folder_ct = os.path.join(base_dir,"tciadataset", "TCIA-NIFTY") # Nifty images folder
        input_folder_seg = os.path.join(base_dir,"tciadataset", "TCIA-SEGM") # Segmentation folder of nifty images
        output_folder_ct = os.path.join(base_dir, "tciadataset","TCIA-PADDED") #unused if dont save images
        output_folder_seg = os.path.join(base_dir,"tciadataset", "TCIA-SEGM-PADDED") #unused if dont save images
        experiment = Experiment(input_folder_ct,output_folder_ct, input_folder_seg,output_folder_seg)
        ct_files = sorted([f for f in os.listdir(input_folder_ct) if f.endswith('.nii') or f.endswith('.nii.gz')])
        seg_files = sorted([f for f in os.listdir(input_folder_seg) if f.endswith('.nii') or f.endswith('.nii.gz')])
        for ct_file, seg_file in zip(ct_files, seg_files):
            ct_path = os.path.join(input_folder_ct, ct_file)
            seg_path = os.path.join(input_folder_seg, seg_file)
            image = experiment.readImage(ct_path)
            mask = experiment.readSegmentation(seg_path)
            resampledImage,resampledSegmentation = process_and_save_images(experiment, image, mask)
            radiomicsObj = experiment.extract_radiomics_mask(resampledImage, resampledSegmentation,"params.yaml")
            stackedFeatureMaps = experiment.stackFeatureMaps(radiomicsObj)
            stackedFeatureMapsInsideImage = experiment.insert_maps_into_mask_bbox(stackedFeatureMaps,resampledSegmentation)
            standardizedFeatureMaps = experiment.pad_or_crop_stacked_feature_maps(stackedFeatureMapsInsideImage).astype(np.float32)
            print("preprocessing is done")
            # Save the standardized feature maps as .npy files
            ctCode = re.findall(r'\d+', ct_file)
            ctCode = ctCode[0] if ctCode else "unknown"
            experiment.saveNumpyArray(
                standardizedFeatureMaps,
                os.path.join("numpyFmaps", "standardized_feature_maps_" + ctCode + ".npz")
            )

         
    except Exception as e:
        print(f"An error occurred: {e}")

# %%
