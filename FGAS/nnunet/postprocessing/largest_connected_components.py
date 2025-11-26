import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label, generate_binary_structure


def keep_largest_connected_component_of_mask(mask_array, connectivity=26):
    binary_mask = (mask_array != 0).astype(np.uint8)

    if np.sum(binary_mask) == 0:
        return mask_array

    if connectivity == 26:
        structure = generate_binary_structure(3, 3)
    elif connectivity == 18:
        structure = np.ones((3, 3, 3))
        structure[0, 0, 0] = 0
        structure[0, 0, 2] = 0
        structure[0, 2, 0] = 0
        structure[0, 2, 2] = 0
        structure[2, 0, 0] = 0
        structure[2, 0, 2] = 0
        structure[2, 2, 0] = 0
        structure[2, 2, 2] = 0
    else:
        structure = generate_binary_structure(3, 1)

    labeled_mask, num_features = label(binary_mask, structure=structure)

    if num_features == 0:
        return mask_array

    component_sizes = np.bincount(labeled_mask.ravel())
    component_sizes[0] = 0
    largest_component_idx = np.argmax(component_sizes)

    largest_component_mask = (labeled_mask == largest_component_idx)
    result_array = np.where(largest_component_mask, mask_array, 0)

    return result_array


def process_mask_keep_largest_component(input_path, output_path, connectivity=26):
    reader = sitk.ImageFileReader()
    reader.SetFileName(input_path)
    image = reader.Execute()
    mask_array = sitk.GetArrayFromImage(image)
    original_dtype = mask_array.dtype

    processed_array = keep_largest_connected_component_of_mask(mask_array, connectivity)

    output_image = sitk.GetImageFromArray(processed_array.astype(original_dtype))
    output_image.CopyInformation(image)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_path)
    writer.Execute(output_image)
    print(f"Post-processing completed, saved to: {output_path}")


def process_all_masks(input_dir, output_dir, connectivity=26):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".nii.gz"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            print(f"Processing file: {filename}")
            process_mask_keep_largest_component(input_path, output_path, connectivity)