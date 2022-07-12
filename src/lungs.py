import SimpleITK as sitk
import numpy as np
from lungmask import mask


def segment_lungs(input_path):

    input_image = sitk.ReadImage(input_path)

    # Make last dimension axial
    image = sitk.GetArrayFromImage(input_image)
    image = np.swapaxes(image, 0, 2)

    segmentation = mask.apply(input_image)
    # Make last dimension axial
    segmentation = np.swapaxes(segmentation, 0, 2)
    background_voxel_value = image.min()

    # if dim_background:
    #     image[segmentation != 0] -= dim_value
    # else:
    image[segmentation == 0] = background_voxel_value

    image = np.swapaxes(image, 0, 2)
    result_out = sitk.GetImageFromArray(image)

    result_out.CopyInformation(input_image)

    return result_out


def generate_segmentation(input_path, output_path):

    result_out = segment_lungs(input_path)

    sitk.WriteImage(result_out, output_path)


if __name__ == "__main__":
    fire.Fire()
