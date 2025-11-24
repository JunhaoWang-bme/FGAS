# Unify dimensions and spatial parameters (images and mask)
import SimpleITK as sitk
import os

mr_folder = r"H:\ExperimentRecord\imagesTr"
mask_folder = r"H:\ExperimentRecord\labelsTr"

for filename in os.listdir(mr_folder):
    if filename.endswith('.nii.gz'):
        mr_path = os.path.join(mr_folder, filename)
        mask_path = os.path.join(mask_folder, filename)

        if os.path.exists(mask_path):
            try:
                mr_image = sitk.ReadImage(mr_path)
                mask_image = sitk.ReadImage(mask_path)

                # spacing, origin, direction
                spacing = mr_image.GetSpacing()
                origin = mr_image.GetOrigin()
                direction = mr_image.GetDirection()

                mask_image.SetSpacing(spacing)
                mask_image.SetOrigin(origin)
                mask_image.SetDirection(direction)

                sitk.WriteImage(mask_image, mask_path)
                print(f"Processed {filename} successfully.")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"Mask file {filename} not found in {mask_folder}.")