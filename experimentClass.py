import os
import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
import pydicom as pdcm
from radiomics import featureextractor
import os

import SimpleITK as sitk
import six

import nrrd
from radiomics import featureextractor



class Experiment:
    def __init__(self, input_folder_ct, output_folder_ct, input_folder_seg, output_folder_seg):
        self.input_folder_ct = input_folder_ct
        self.output_folder_ct = output_folder_ct
        self.input_folder_seg = input_folder_seg
        self.output_folder_seg = output_folder_seg
        self.reader = sitk.ImageSeriesReader()
        self.nifty_images = None
        self.nifty_segmentations = None
        self.originaImageSizes = []
        self.originalVoxelSizes = []
        self.resampledImageSizes = []
        self.imagePaths = []
        self.segmentationPaths = []
        self.imagesWidth = []
        self.imagesHeight = []
        self.imagesDepth = []
        self.outputFolderFeatureMaps = os.path.join("", "padded_feature_maps_inside_image")
        self.index=0
        
    def convert_dicom_to_nifti(self, dicom_folder, output_file):
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

        
    def process_dicom_series(self, input_folder, output_folder):
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
                                    # Optionally: self.debug_dicom_series(sub_series_path)
                                    output_file = os.path.join(output_folder, f"{patient_folder}.nii.gz")
                                    self.convert_dicom_to_nifti(sub_series_path, output_file)
                                except RuntimeError as e:
                                    print(f"Error processing folder {sub_series_path}: {e}")
    
        # Conversion and saving are handled in convert_dicom_to_nifti.
    def setImagePaths(self, imagePaths):
        """
        Sets the image paths for the experiment.
        """
        for root, _, files in os.walk(self.input_folder_ct):
            for file in files:
                self.imagePaths.append(os.path.join(root, file))

    def setSegmentationPaths(self, segmentationPaths):
        """
        Sets the segmentation paths for the experiment.
        """
        for root, _, files in os.walk(self.input_folder_seg):
            for file in files:
                self.segmentationPaths.append(os.path.join(root, file))
    def readImage(self, image_folder):
        """
        Reads a nifty file from a path and returns the image.
        """

        image = sitk.ReadImage(image_folder)
        return image
        
    def readSegmentation(self, seg_folder):
        """
        Reads a nifty file from a path and returns the image.
        """

        seg = sitk.ReadImage(seg_folder)
        return seg
    
    def isotropicResampleImage(self, image, new_spacing=(1.0, 1.0, 1.0)):
        """
        Resamples the image to isotropic spacing.
        """
        print(self.getPixelId(image),"get pixel`")
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        
        new_size = [
            int(np.round(original_size[i] * (original_spacing[i] / new_spacing[i])))
            for i in range(3)
        ]
        
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetOutputSpacing(new_spacing)
        resample_filter.SetSize(new_size)
        resample_filter.SetOutputDirection(image.GetDirection())
        resample_filter.SetOutputOrigin(image.GetOrigin())
        
        resampled_image = resample_filter.Execute(image)
        
        return resampled_image
    
    def isotropicResampleSegmentation(self, seg, new_spacing=(1.0, 1.0, 1.0)):
        """
        Resamples the segmentation to isotropic spacing.
        """
        original_spacing = seg.GetSpacing()
        original_size = seg.GetSize()
        
        new_size = [
            int(np.round(original_size[i] * (original_spacing[i] / new_spacing[i])))
            for i in range(3)
        ]
        self.imagesHeight.append(new_size[1])
        self.imagesWidth.append(new_size[0])
        self.imagesDepth.append(new_size[2])
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetOutputSpacing(new_spacing)
        resample_filter.SetSize(new_size)
        resample_filter.SetOutputDirection(seg.GetDirection())
        resample_filter.SetOutputOrigin(seg.GetOrigin())
        resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)  # Use nearest neighbor for segmentation
        
        resampled_segmentation = resample_filter.Execute(seg)
        
        return resampled_segmentation
    
    def hounsfieldWindowing(self, image, window_center=40, window_width=400):
        """
        Applies Hounsfield windowing to the image. Defaults for abdominal CT.
        Returns a SimpleITK image.
        """
        original_image = image  # Save the original SimpleITK image
        arr = sitk.GetArrayFromImage(image)
        arr = np.clip(arr, window_center - window_width // 2, window_center + window_width // 2)
        arr = (arr - (window_center - window_width // 2)) / window_width * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint16)
        out_img = sitk.GetImageFromArray(arr)
        out_img.CopyInformation(original_image)
        return out_img
    
    def storeImage(self, image, output_folder):
        """
        Saves the image to a folder.
        """
        print(f"Saving image to '{output_folder}'")

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_folder)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        try:
            sitk.WriteImage(image, output_folder)
        except RuntimeError as e:
            print(f"Error saving image to '{output_folder}': {e}")
            return False
        print(f"Saved image to '{output_folder}'")

    def maxDimensions(self):
        """
        Returns the maximum dimensions between the images and segmentations.
        """
        max_width = max(self.imagesWidth)
        max_height = max(self.imagesHeight)
        max_depth = max(self.imagesDepth)
        print(f"Max dimensions: {max_width} x {max_height} x {max_depth}")
    def getPixelId(self, image):
        """
        Returns the pixel ID of the image. 2 = int16
        """
        return image.GetPixelID()
    
    # def pad_feature_map(self, feature_map, target_shape):
    #     """
    #     Pads a 3D feature map with zeros.
    #     XY are padded symmetrically.
    #     Z is padded at the end only (bottom).
    #     """
    #     current_shape = feature_map.shape  # (Z, Y, X)

    #     pad_z = (0, max(0, target_shape[0] - current_shape[0]))  # pad only after Z
    #     pad_y = (
    #         (max(0, (target_shape[1] - current_shape[1]) // 2),
    #          max(0, (target_shape[1] - current_shape[1] + 1) // 2))
    #     )
    #     pad_x = (
    #         (max(0, (target_shape[2] - current_shape[2]) // 2),
    #          max(0, (target_shape[2] - current_shape[2] + 1) // 2))
    #     )

    #     padded = np.pad(feature_map, (pad_z, pad_y, pad_x), mode='constant', constant_values=0)
    #     return padded
    
    def extract_radiomics_mask(self, image, mask,params_path, voxel_based=True):
        """
        Extracts radiomics features from the given image without a mask.
        Returns a dictionary of feature maps (if voxel_based=True) or feature values.
        """
        extractor = featureextractor.RadiomicsFeatureExtractor(params_path)
        print("extracting features")
        radiomicFmapsObj = extractor.execute(image,mask, voxelBased=voxel_based)
        return radiomicFmapsObj
    
    def save_feature_maps(self, feature_maps, prefix="feature"):
        """
        Saves all feature maps (SimpleITK images) in the given dictionary to the output directory.
        Only saves items that are SimpleITK images.
        """
        if not os.path.exists(self.outputFolderFeatureMaps):
            os.makedirs(self.outputFolderFeatureMaps)
        for key, val in feature_maps.items():
            if isinstance(val, sitk.Image):
                filename = os.path.join(self.outputFolderFeatureMaps, f"{prefix}_{key}_{self.index}.nrrd")
                sitk.WriteImage(val, filename)
                print(f"Saved feature map: {filename}")
        self.index+=1

    def stackFeatureMaps(self, featureMapsObj):
        """
        Stacks all feature maps (SimpleITK images) in the given dictionary into a single 3D image.
        """
        feature_map_list = []
        for key, val in featureMapsObj.items():
            if isinstance(val, sitk.Image):
                feature_map_list.append(sitk.GetArrayFromImage(val))
        stacked_feature_maps = np.stack(feature_map_list, axis=0)
        return stacked_feature_maps
    
    
    def conver_nrrd_to_nifti(self, nrrd_path):
        """
        Converts a NRRD file to NIfTI format.
        """
        data, header = nrrd.read(nrrd_path)
        img = sitk.GetImageFromArray(data)
        # Fix spacing/origin/direction assignment
        if 'space directions' in header:
            spacing = [float(np.linalg.norm(v)) for v in header['space directions']]
            img.SetSpacing(tuple(spacing))
        if 'space origin' in header:
            img.SetOrigin(tuple(header['space origin']))
        # Optionally set direction if available
        output_dir = "nifti"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, nrrd_path.split("/")[-1].replace(".nrrd", ".nii.gz"))
        sitk.WriteImage(img, output_path)

    def compareNiftyMapsWithNiftyMask(self, nifty_map_path, nifty_mask_path):
        """
        Compares a NIfTI map with a NIfTI mask.
        """
        print(f"Comparing {nifty_map_path} with {nifty_mask_path}")
        nifty_map = sitk.ReadImage(nifty_map_path)
        nifty_mask = sitk.ReadImage(nifty_mask_path)
        nifty_map_arr = sitk.GetArrayFromImage(nifty_map)
        nifty_mask_arr = sitk.GetArrayFromImage(nifty_mask)
        # Use LabelShapeStatisticsImageFilter to get bounding box
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute(nifty_mask)
        bounding_box = label_shape_filter.GetBoundingBox(1)
        featureMapInImage = self.insert_map_into_mask_bbox(nifty_map, nifty_mask)
        output_path = os.path.join(self.outputFolderFeatureMaps, f"feature_map_in_image_ClusterProminence_{self.index}.nii.gz")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.storeImage(featureMapInImage, output_path)
        print(f"Map shape: {nifty_map_arr.shape}")
        print(f"Mask shape: {nifty_mask_arr.shape}")
        print(f"Map pixel type: {nifty_map.GetPixelID()}")
        print(f"Mask pixel type: {nifty_mask.GetPixelID()}")
        print(f"Mask Bounding Box: {bounding_box}")
        self.index+=1

    def boundingBoxOfSegmentation(self, seg_path):
        """
        Returns the bounding box of the segmentation.
        """
        seg = sitk.ReadImage(seg_path)
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute(seg)
        bounding_box = label_shape_filter.GetBoundingBox(1) 
        print(f"Bounding box: {bounding_box.GetSize()}")
        return bounding_box
    
    def insert_map_into_mask_bbox(self, map_img, mask_img):
        """
        Inserts the map (feature map) into a zero array with the same shape as the mask,
        starting at the mask bounding box indices.
        """
        # Get arrays
        mask_arr = sitk.GetArrayFromImage(mask_img)
        map_arr = sitk.GetArrayFromImage(map_img)

        # Get mask bounding box (x_min, y_min, z_min, size_x, size_y, size_z)
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute(mask_img)
        bbox = label_shape_filter.GetBoundingBox(1)
        x_min, y_min, z_min, size_x, size_y, size_z = bbox

        # SimpleITK arrays are (z, y, x)
        # So, for numpy: z_min, y_min, x_min
        z_min_np, y_min_np, x_min_np = z_min, y_min, x_min

        # Calculate the region to insert (use the minimum of both shapes)
        insert_shape = (
            min(size_z, map_arr.shape[0]),
            min(size_y, map_arr.shape[1]),
            min(size_x, map_arr.shape[2])
        )

        # Crop the map if needed
        cropped_map = map_arr[:insert_shape[0], :insert_shape[1], :insert_shape[2]]

        # Make sure we don't go out of bounds in the mask array
        z_end = z_min_np + insert_shape[0]
        y_end = y_min_np + insert_shape[1]
        x_end = x_min_np + insert_shape[2]

        # Insert the cropped map into the mask-shaped array
        output_arr = np.zeros_like(mask_arr, dtype=map_arr.dtype)
        output_arr[z_min_np:z_end, y_min_np:y_end, x_min_np:x_end] = cropped_map

        output_img = sitk.GetImageFromArray(output_arr)
        output_img.CopyInformation(mask_img)
        
        return output_img
    
    def insert_maps_into_mask_bbox(self, stacked_maps, mask_img):
        """
        Inserts each channel of stacked_maps (shape: [C, Z, Y, X]) into a zero array
        with the same shape as the mask, starting at the mask bounding box indices.
        Returns a 4D array: (C, Z, Y, X)
        """
        mask_arr = sitk.GetArrayFromImage(mask_img)
        num_channels = stacked_maps.shape[0]

        # Get mask bounding box (x_min, y_min, z_min, size_x, size_y, size_z)
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute(mask_img)
        bbox = label_shape_filter.GetBoundingBox(1)
        x_min, y_min, z_min, size_x, size_y, size_z = bbox
        z_min_np, y_min_np, x_min_np = z_min, y_min, x_min

        # Prepare output array
        output_arr = np.zeros((num_channels,) + mask_arr.shape, dtype=stacked_maps.dtype)

        for c in range(num_channels):
            fmap = stacked_maps[c]
            # Insert into the bounding box region
            output_arr[c,
                z_min_np:z_min_np+fmap.shape[0],
                y_min_np:y_min_np+fmap.shape[1],
                x_min_np:x_min_np+fmap.shape[2]
            ] = fmap

        return output_arr
    
    def pad_or_crop_image_to_shape(self, image, target_shape=(500, 500, 310)):
        """
        Pads or crops a SimpleITK image to the specified target shape.
        The target_shape should be in (X, Y, Z) order to match SimpleITK's GetSize().
        """
        arr = sitk.GetArrayFromImage(image)  # (Z, Y, X)
        # Convert target_shape (X, Y, Z) to (Z, Y, X) for numpy
        target_shape_np = (target_shape[2], target_shape[1], target_shape[0])
        current_shape = arr.shape

        # Calculate padding for each dimension
        pad_z = max(0, target_shape_np[0] - current_shape[0])
        pad_y = max(0, target_shape_np[1] - current_shape[1])
        pad_x = max(0, target_shape_np[2] - current_shape[2])

        # Pad as (before, after) for each axis
        pad_width = (
            (0, pad_z),
            (0, pad_y),
            (0, pad_x)
        )
        arr_padded = np.pad(arr, pad_width, mode='constant', constant_values=0)

        # Crop if needed
        arr_padded = arr_padded[:target_shape_np[0], :target_shape_np[1], :target_shape_np[2]]

        # Convert back to SimpleITK image
        padded_img = sitk.GetImageFromArray(arr_padded)
        padded_img.SetSpacing(image.GetSpacing())
        padded_img.SetOrigin(image.GetOrigin())
        padded_img.SetDirection(image.GetDirection())
        return padded_img
    
    def pad_or_crop_stacked_feature_maps(self, stacked_feature_maps, target_shape=(500, 500, 310)):
        """
        Pads or crops each channel of a stacked feature map array (shape: [C, Z, Y, X])
        to the specified target shape (X, Y, Z). Returns a new stacked array.
        """
        num_channels = stacked_feature_maps.shape[0]
        # Convert target_shape (X, Y, Z) to (Z, Y, X) for numpy
        target_shape_np = (target_shape[2], target_shape[1], target_shape[0])
        padded_stack = np.zeros((num_channels,) + target_shape_np, dtype=stacked_feature_maps.dtype)

        for c in range(num_channels):
            arr = stacked_feature_maps[c]
            current_shape = arr.shape

            pad_z = max(0, target_shape_np[0] - current_shape[0])
            pad_y = max(0, target_shape_np[1] - current_shape[1])
            pad_x = max(0, target_shape_np[2] - current_shape[2])

            pad_width = (
                (0, pad_z),
                (0, pad_y),
                (0, pad_x)
            )
            arr_padded = np.pad(arr, pad_width, mode='constant', constant_values=0)
            arr_padded = arr_padded[:target_shape_np[0], :target_shape_np[1], :target_shape_np[2]]

            padded_stack[c] = arr_padded

        return padded_stack
    
   
