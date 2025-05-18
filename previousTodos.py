#%%
import os
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
def debug_dicom_series(dicom_folder):
    """
    Debug function to check if DICOM series are detected in a folder.
    """
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_folder)
    if not series_ids:
        print(f"No DICOM series found in folder: {dicom_folder}")
    else:
        print(f"Found DICOM series IDs: {series_ids}")
        for series_id in series_ids:
            dicom_files = reader.GetGDCMSeriesFileNames(dicom_folder, series_id)
            print(f"Series ID: {series_id}, Number of files: {len(dicom_files)}")


def convert_dicom_to_nifti(dicom_folder, output_file):
    """
    Converts a DICOM series to a single .nii.gz file.
    """
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_folder)
    
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in folder: {dicom_folder}")
    
    # Use the first series ID
    series_id = series_ids[0]
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_folder, series_id)
    reader.SetFileNames(dicom_files)
    
    # Load the DICOM series as a 3D volume
    image = reader.Execute()
    
    # Save the image as .nii.gz
    sitk.WriteImage(image, output_file)
    print(f"Converted DICOM series in '{dicom_folder}' to '{output_file}'")


def hounsfield_windowing(image, window_width=400, window_center=40):
    """
    Applies Hounsfield windowing to a CT image.
    """
    image = np.clip(image, window_center - window_width // 2, window_center + window_width // 2)
    image = (image - (window_center - window_width // 2)) / window_width * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def process_dicom_series(input_folder, output_folder):
    """
    Processes all DICOM series in the input folder and converts them to .nii.gz files.
    Handles additional nested folder structures.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    for patient_folder in os.listdir(input_folder):
        patient_path = os.path.join(input_folder, patient_folder)
        if os.path.isdir(patient_path):
            for series_folder in os.listdir(patient_path):
                series_path = os.path.join(patient_path, series_folder)
                if os.path.isdir(series_path):
                    for sub_series_folder in os.listdir(series_path):  # Additional nested loop
                        sub_series_path = os.path.join(series_path, sub_series_folder)
                        if os.path.isdir(sub_series_path):
                            try:
                                debug_dicom_series(sub_series_path)  # Debug the series
                                output_file = os.path.join(output_folder, f"{patient_folder}.nii.gz")
                                convert_dicom_to_nifti(sub_series_path, output_file)
                            except RuntimeError as e:
                                print(f"Error processing folder {sub_series_path}: {e}")

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
      


#todo: aurio na epanaferw mask based feature extraction + epanatopothetisi sto original image
# def extact_features(exp:Experiment,image_filepath:str,params_filepath:str):
#     feature_maps=exp.extract_radiomics_no_mask(image_filepath,params_filepath)
#     exp.save_feature_maps(feature_maps)
# Main script
 #%%
if __name__ == "__main__":
    try:
       
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        input_folder_ct = os.path.join(base_dir,"tciadataset", "TCIA-NIFTY")
        input_folder_seg = os.path.join(base_dir,"tciadataset", "TCIA-SEGM")
        output_folder_ct = os.path.join(base_dir, "tciadataset","TCIA-PADDED")
        output_folder_seg = os.path.join(base_dir,"tciadataset", "TCIA-SEGM-PADDED")
        experiment = Experiment(input_folder_ct,output_folder_ct, input_folder_seg,output_folder_seg)
        # experiment.setImagePaths(input_folder_ct)
        # experiment.setSegmentationPaths(input_folder_seg)
        # process_and_save_images(experiment, output_folder_ct, output_folder_seg)
        # experiment.maxDimensions()
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
            standardizedFeatureMaps = experiment.pad_or_crop_stacked_feature_maps(stackedFeatureMapsInsideImage)
            print("preprocessing is done")

            # print("i stacked feature maps")
            # for i, fmap in enumerate(stackedFeatureMaps):
            #     print(f"Feature map {i}: shape={fmap.shape}, min={np.nanmin(fmap)}, max={np.nanmax(fmap)}, nan count={np.isnan(fmap).sum()}")

            # fMaps=experiment.extract_radiomics_mask(image, mask,"params.yaml")
            # experiment.save_feature_maps(fMaps)
        # feature_maps = sorted([f for f in os.listdir(experiment.outputFolderFeatureMaps) if f.endswith('.nrrd') or f.endswith('.nrrd.gz')])
        # masks = sorted([f for f in os.listdir(output_folder_seg)  if f.endswith('.nii.gz')])
        # feature_maps_nifty = sorted([f for f in os.listdir("nifti") if "ClusterProminence" in f])
        # mapsInsideImagePath = os.path.join("", "feature_maps_inside_image")
        # maps = sorted([f for f in os.listdir(mapsInsideImagePath) if f.endswith('.nii.gz')])
        # padded_fMaps = sorted([f for f in os.listdir(experiment.outputFolderFeatureMaps) if f.endswith('.nii.gz')])
        # print("padded_fMaps length", len(padded_fMaps))
        # for ct_file, seg_file in zip(ct_files, seg_files):
        #     ct_path = os.path.join(output_folder_ct, ct_file)
        #     seg_path = os.path.join(output_folder_seg, seg_file)
        #     image = experiment.readImage(ct_path)
        #     mask = experiment.readSegmentation(seg_path)
        #     fMaps=experiment.extract_radiomics_mask(image, mask,"params.yaml")
        #     experiment.save_feature_maps(fMaps)
        
        # for feature_map in feature_maps:
        #     feature_map_path = os.path.join(experiment.outputFolderFeatureMaps, feature_map)
        #     experiment.conver_nrrd_to_nifti(feature_map_path)
        # for mask, feature_map in zip(masks, feature_maps_nifty):
        #     mask_path = os.path.join(output_folder_seg, mask)
        #     feature_map_path = os.path.join(os.path.join(os.getcwd(),"nifti"), feature_map)
        #     experiment.compareNiftyMapsWithNiftyMask(feature_map_path,mask_path)

        # for map in maps:
        #     map_path = os.path.join(mapsInsideImagePath, map)
        #     image  = experiment.readImage(map_path)
        #     paddedFeatureMapInImage = experiment.pad_or_crop_image_to_shape(image)
        #     experiment.storeImage(paddedFeatureMapInImage, os.path.join(experiment.outputFolderFeatureMaps, f"padded_cluster_prominence{experiment.index}.nii.gz"))
        #     experiment.index+=1

        # for padded_fMap in padded_fMaps:
        #     padded_fMap_path = os.path.join(experiment.outputFolderFeatureMaps, padded_fMap)
        #     imageSize = experiment.readImage(padded_fMap_path).GetSize()
        #     print(f"Image size of {padded_fMap}: {imageSize}")

    except Exception as e:
        print(f"An error occurred: {e}")
    # parent_folder = r"D:\BME-AUTH\Courses\DiplomaThesis\Material\Code\voxel-based-fmaps\original_glcm_JointEntropy_1.nrrd"
    # data,header = nrrd.read(parent_folder)
    # print(data.shape)
    # print(header)

# %%
