import SimpleITK as sitk
import os
from glob import glob

SAG_DIR = "./sag_data"
COR_TRA_DIR = "./cor_tra_data"
MASK_DIR = "./mask_data"
OUTPUT_TRANSFORM_DIR = "./transforms"
OUTPUT_MASK_DIR = "./updated_masks"

# Create output directories
os.makedirs(OUTPUT_TRANSFORM_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)


def register_images_batch(fixed_image_path, moving_image_path, output_transform_path):
    fixed_image = sitk.ReadImage(fixed_image_path)
    moving_image = sitk.ReadImage(moving_image_path)

    print(f"\n=== Registration: {os.path.basename(moving_image_path)} ===")
    print(f"Fixed (sag) Direction: {fixed_image.GetDirection()[:6]}...")
    print(f"Moving Direction: {moving_image.GetDirection()[:6]}...")
    print(f"Fixed (sag) Spacing: {fixed_image.GetSpacing()}")
    print(f"Moving Spacing: {moving_image.GetSpacing()}")

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.01)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-6, numberOfIterations=200,
        gradientMagnitudeTolerance=1e-8
    )
    registration.SetOptimizerScalesFromPhysicalShift()

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, moving_image, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration.SetInitialTransform(initial_transform, inPlace=False)
    registration.SetInterpolator(sitk.sitkLinear)

    final_transform = registration.Execute(fixed_image, moving_image)
    sitk.WriteTransform(final_transform, output_transform_path)
    print(f"Registration completed, transform saved to: {output_transform_path}")
    return final_transform


def apply_transform_to_mask_batch(fixed_image_path, mask_path, transform_path, output_mask_path):
    fixed_image = sitk.ReadImage(fixed_image_path)
    mask_image = sitk.ReadImage(mask_path)
    transform = sitk.ReadTransform(transform_path)

    print(f"\n=== Processing mask: {os.path.basename(mask_path)} ===")
    print(f"Fixed (sag) Direction: {fixed_image.GetDirection()[:6]}...")
    print(f"Original Mask Direction: {mask_image.GetDirection()[:6]}...")

    mask_image.SetDirection(fixed_image.GetDirection())

    resampled_mask = sitk.Resample(
        mask_image, fixed_image, transform,
        sitk.sitkNearestNeighbor, 0.0, mask_image.GetPixelID()
    )

    resampled_mask.SetDirection(fixed_image.GetDirection())
    resampled_mask.SetSpacing(fixed_image.GetSpacing())
    resampled_mask.SetOrigin(fixed_image.GetOrigin())

    sitk.WriteImage(resampled_mask, output_mask_path)
    print(f"Mask updated, saved to: {output_mask_path}")
    return resampled_mask


def batch_process():
    sag_files = glob(os.path.join(SAG_DIR, "*sag*.nii.gz"))
    if not sag_files:
        raise FileNotFoundError(f"No sag fixed image found in {SAG_DIR}")
    fixed_sag_path = sag_files[0]
    print(f"Fixed image (sag): {fixed_sag_path}")

    moving_files = glob(os.path.join(COR_TRA_DIR, "*cor*.nii.gz")) + glob(os.path.join(COR_TRA_DIR, "*tra*.nii.gz"))
    if not moving_files:
        raise FileNotFoundError(f"No cor/tra moving images found in {COR_TRA_DIR}")

    for moving_path in moving_files:
        img_id = os.path.basename(moving_path).replace(".nii.gz", "").replace("cor", "").replace("tra", "").strip("_")

        transform_path = os.path.join(OUTPUT_TRANSFORM_DIR, f"{img_id}_to_sag.tfm")
        register_images_batch(fixed_sag_path, moving_path, transform_path)

        mask_pattern = os.path.join(MASK_DIR, f"*{img_id}*.nii.gz")
        mask_files = glob(mask_pattern)
        if not mask_files:
            print(f"Warning: No mask found for {img_id}, skipping mask processing")
            continue
        mask_path = mask_files[0]

        output_mask_path = os.path.join(OUTPUT_MASK_DIR, f"{img_id}_mask_updated.nii.gz")
        apply_transform_to_mask_batch(fixed_sag_path, mask_path, transform_path, output_mask_path)

    print("\n=== Batch processing completed ===")


if __name__ == "__main__":
    batch_process()