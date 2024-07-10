
# Localization results for `ChangeThisTEMPORARY`
This directory contains the localization predictions for the image found at `data/tmp/20230908001619-00-06.jpg`. For some pipelines, this path may be non-standard, please confer with the relevant developer for clarification.

## Files
The predictions are saved in two formats: .pt (`PyTorch` pickle) and .json (JSON).
The `PyTorch` file contains the pickled `TensorPredictions` object dictionary, while the JSON file contains the data serialized in a more human-readable format, which can reasonably be deserialized by anyone not familiar with the `TensorPredictions` object using any programming language with access to basic JSON libraries.

### JSON format
The JSON file contains the following data:
> - **`boxes`** (list of lists of integers):  
    The bounding boxes for each prediction in the format [x1, y1, x2, y2], where (x1, y1) is the bottom left corner and (x2, y2) is the top right corner. \
    Coordinates are given in the "image pixel coordinate system".

> - **`contours`** (list of lists of lists of floats or integers):  
    The contours for each prediction in the format [[x1, x2, ..., xn], [y1, y2, ..., yn]], where (x1, y1) is the first point, (x2, y2) is the second point, and so on. \
    Coordinates are given in the "mask coordinate system" which is approximately proportional to the "image coordinate system". \
    Points should be ordered in clockwise order, if not please contact the developers. 

> - **`confs`** (list of floats):  
    The confidences for each prediction.

> - **`classes`** (list of integers):  
    The indices of the classes for each prediction.

> - **`scales`** (list of floats):  
    The scale at which a given prediction was found. A smaller scale corresponds to a more zoomed-out view of the image.

> - **`identifier`** (string):  
    An identifier for the predictions.

> - **`image_path`** (string):  
    The path to the image that the predictions are for. May be non-standard.

> - **`image_width`** (integer):  
    The width of the image that the predictions are for.

> - **`image_height`** (integer):  
    The height of the image that the predictions are for.

> - **`mask_width`** (integer):  
    The width of the masks, where the contours are derived from.

> - **`mask_height`** (integer):  
    The height of the masks, where the contours are derived from.

The mask coordinates are given in the mask coordinate system, therefore they must be scaled by the ratio between the image and the mask to get the image coordinates: (unless the mask dimensions are equal to the image dimensions)

```
image_x = mask_x * (image_width / mask_width)
image_y = mask_y * (image_height / mask_height)
```

The bounding box coordinates are given in the image coordinate system, so they do not need to be scaled to be used in the image.

### Image Coordinate System
The image coordinate system is simply the integer pixel coordinate system of the image, where the **top left** corner is (`0`; `0`) and the **bottom right** corner is (`image_width`; `image_height`).

## Deserializations
The .pt pickle file can be deserialized into a `TensorPredictions` object using either `torch.load("data/tmp_fb/metadata/metadata_20230908001619-00-06_UUID_ChangeThisTEMPORARY.pt")` `TensorPredictions().load("data/tmp_fb/metadata/metadata_20230908001619-00-06_UUID_ChangeThisTEMPORARY.pt")`. OBS: This may be deprecated in the future, since the .json file contains the same data in a more human-readable format, and serialization/deserialization is reasonably fast.

The JSON can be deserialized into a `TensorPredictions` object using `TensorPredictions().load("data/tmp_fb/metadata/metadata_20230908001619-00-06_UUID_ChangeThisTEMPORARY.json")`.