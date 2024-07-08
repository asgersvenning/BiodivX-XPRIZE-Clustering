import os, re, shutil, json, csv, pickle, argparse, glob

from typing import Callable, List, Tuple, Dict, Union, Any, Optional, Iterable

from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from urllib.request import urlretrieve

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

from statistics import mode
from tqdm import tqdm
from PIL import Image

import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
from xprize_insectnet.hierarchical.model import model_from_state_file

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity

ERDA_MODEL_ZOO_TEMPLATE = "https://anon.erda.au.dk/share_redirect/aRbj0NCBkf/{}"
STANDARD_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Input parsing 
IMG_REGEX = re.compile(r'\.(jp[e]{0,1}g|png|dng)$', re.IGNORECASE)

def is_image(file_path):
    return bool(re.search(IMG_REGEX, file_path)) and os.path.isfile(file_path)

def is_txt(file_path):
    return file_path.endswith('.txt') and os.path.isfile(file_path)

def is_dir(file_path):
    return os.path.isdir(file_path)

def is_glob(file_path):
    return not (is_image(file_path) or is_dir(file_path))

def type_of_path(file_path):
    if is_image(file_path):
        return 'image'
    elif is_txt(file_path):
        return 'txt'
    elif is_dir(file_path):
        return 'dir'
    elif is_glob(file_path):
        return 'glob'
    else:
        return 'unknown'

def get_images(input_path_dir_globs : Union[str, List[str]]) -> List[str]:
    if isinstance(input_path_dir_globs, str):
        input_path_dir_globs = [input_path_dir_globs]
    images = []
    for path in input_path_dir_globs:
        match type_of_path(path):
            case 'image':
                images.append(path)
            case 'txt':
                with open(path, 'r') as f:
                    paths = [path.strip() for path in f.readlines() if len(path.strip()) > 0]
                images.extend(get_images(paths))
            case 'dir':
                images.extend(glob.glob(os.path.join(path, '*')))
            case 'glob':
                images.extend(glob.glob(path))
            case _:
                raise ValueError(f"Unknown path type: {path}")
    if len(images) == 0:
        raise ValueError("No images found")
    return images

# Time stamp extraction
# def get_timestamp(image_path : Union[str, Iterable[str]], time_format : str) -> str:
#     if not isinstance(image_path, str) and not all(map(lambda p : isinstance(p, str), image_path)):
#         raise TypeError(f"`image_path` must be a string, got {type(image_path)}")
#     if not isinstance(time_format, str):
#         raise TypeError(f"`time_format` must be a string, got {type(time_format)}")

#     time_regex_parts = re.findall(r'(%([a-zA-Z])\2+)', time_format)
#     time_regex_parts = {s : len(part) - 1 for part, s in time_regex_parts}
#     time_regex_sep = re.split("|".join("%" + s * l for s, l in time_regex_parts.items()), time_format)
#     time_regex_format = "{}".join(time_regex_sep)
#     time_regex = time_regex_format.format(*[f'(\d{{{p}}})' for p in time_regex_parts.values()])

#     predefined_order = {"Y" : 0, "m" : 1, "d" : 2, "H" : 3, "M" : 4, "S" : 5}
#     defaults = {"Y" : "0000", "m" : "00", "d" : "00", "H" : "00", "M" : "00", "S" : "00"}
#     defaults = {predefined_order[k] : v for k, v in defaults.items()}
#     reorder = {i : predefined_order[k] for i, k in enumerate(time_regex_parts.keys())}

#     if isinstance(image_path, str):
#         time_parts = get_matches_in_order(image_path, time_regex, reorder, defaults)
#         return format_timestamp(time_parts)
#     elif isinstance(image_path, (list, tuple)):
#         time_parts = [get_matches_in_order(path, time_regex, reorder, defaults) for path in image_path]
#         return [format_timestamp(parts) for parts in time_parts]
#     else:
#         raise TypeError(f"`image_path` must be a string, list or tuple, got {type(image_path)}")
    
# def get_matches_in_order(image_path : str, time_regex : str, reorder : Dict[int, int], default_values : Dict[int, str]) -> str:
#     if not isinstance(image_path, str):
#         raise TypeError(f"`image_path` must be a string, got {type(image_path)}")
#     if not isinstance(time_regex, str):
#         raise TypeError(f"`time_regex` must be a string, got {type(time_regex)}")
#     if not isinstance(reorder, dict):
#         raise TypeError(f"`reorder` must be a dictionary, got {type(reorder)}")
#     if not isinstance(default_values, dict):
#         raise TypeError(f"`default_values` must be a dictionary, got {type(default_values)}")
#     image_path = os.path.basename(image_path)
#     matches = re.search(time_regex, image_path)
#     if matches is None:
#         raise ValueError(f"No matches found in '{image_path}' using '{time_regex}'")
#     matches = matches.groups()
#     matches = {i : matches[i] for i in reorder.values()}
#     for i, default in default_values.items():
#         if i not in matches:
#             matches[i] = default
#     matches = [matches[i] for i in range(len(matches))]
#     return matches

# def format_timestamp(time_parts : str, time_format : str = "{}-{}-{} {}:{}:{}") -> str:
#     return time_format.format(*time_parts)

def convert_timeformat(time_format : str) -> str:
    # Pattern to find and replace date components
    patterns = {
        r'%Y{2,4}': '%Y',  # Year
        r'%m{1,2}': '%m',  # Month
        r'%d{1,2}': '%d',  # Day
        r'%H{1,2}': '%H',  # Hour
        r'%M{1,2}': '%M',  # Minute
        r'%S{1,2}': '%S'   # Second
    }
    
    # Perform the replacements
    for pattern, replacement in patterns.items():
        time_format = re.sub(pattern, replacement, time_format)
    
    return time_format

def create_regex_from_format(format_string : str):
    # Define the mapping from format directive to regex pattern
    format_mappings = {
        "%Y": r"(?P<Y>\d{4})",
        "%m": r"(?P<m>\d{2})",
        "%d": r"(?P<d>\d{2})",
        "%H": r"(?P<H>\d{2})",
        "%M": r"(?P<M>\d{2})",
        "%S": r"(?P<S>\d{2})"
    }
    
    # Escape characters that are not format directives
    escaped_format = re.escape(format_string)
    
    # Replace format directives with corresponding regex patterns
    for directive, pattern in format_mappings.items():
        escaped_format = escaped_format.replace(re.escape(directive), pattern)
    
    return escaped_format

def parse_time_string(time_string : str, format_string : str):
    # Create regex pattern from format string
    format_string = convert_timeformat(format_string)
    regex_pattern = create_regex_from_format(format_string)
    
    # Match the time string with the regex pattern
    match = re.search(regex_pattern, time_string)
    if match:
        time_elements = match.groupdict()
        print(time_elements, match.group(0))
        try:
            return datetime.strptime(match.group(0), format_string)
        except ValueError as e:
            raise ValueError(f"Time string '{time_string}' does not match format '{format_string}'")
    
    raise ValueError(f"Time string '{time_string}' does not match the pattern derived from format '{format_string}'")

def search_and_convert_timestamp(image_path : str, regex_pattern : Union[str, re.Pattern], from_format : str, to_format : str) -> str:
    match = re.search(regex_pattern, image_path)
    if match:
        original_datetime = datetime.strptime(match.group(0), from_format)
        target_datetime_str = original_datetime.strftime(to_format)
        return target_datetime_str
    else:
        raise ValueError(f"Time string '{image_path}' does not match the pattern derived from format '{from_format}'")

# Time stamp extraction
def get_timestamp(image_path : Union[str, Iterable[str]], time_format : str, default_time_format : str = "%Y-%m-%d %H:%M:%S") -> str:
    if not isinstance(image_path, str) and not all(map(lambda p : isinstance(p, str), image_path)):
        raise TypeError(f"`image_path` must be a string, got {type(image_path)}")
    if not isinstance(time_format, str):
        raise TypeError(f"`time_format` must be a string, got {type(time_format)}")

    time_format = convert_timeformat(time_format)
    regex_pattern = create_regex_from_format(time_format)

    if isinstance(image_path, str):
        return search_and_convert_timestamp(image_path, regex_pattern, time_format, default_time_format)
    elif isinstance(image_path, (list, tuple)):
        return [search_and_convert_timestamp(p, regex_pattern, time_format, default_time_format) for p in image_path]
    else:
        raise TypeError(f"`image_path` must be a string, list or tuple, got {type(image_path)}")

#----------------------------------------------------------------------------
# Embedding base class

class Embedding:
    """Abstract class to manage inputs (images crops) and outputs (embeddings).

    The class method `self.compute` is abstract and must be overridden in child-classes.
    """

    def __init__(self,
                 filenames: list = None,
                 model_path: str = None,
                 device: str = "cpu",
                 **kwargs,
                 ) -> None:

        self.model_path = model_path

        # Device on which computations will be done
        self.device = device

        # Array of embeddings
        self.embs = None

        # Labels of the embeddings
        # Can stay None if no labels exist
        self.labels = None

        # List of filenames of the images
        if filenames is None:
            raise ValueError("Filenames must be provided.")
        self.filenames = filenames

    def save(self, out_file : str, keep_parent_dir: bool = True) -> None: # TODO: allows to save absolute path of images instead of relative.
        """Save the embeddings.

        Currently accepted file format: pkl and json.

        Parameters
        ----------
        out_file : str
            Path of the output file where to store the embeddings.
            File extension must be `.pkl` or `json`.
        """
        out_path = Path(out_file)
        valid_formats = [".json", ".pkl"]
        assert out_path.suffix.lower() in valid_formats, "[Error] Output file does not have a valid format: {}".format(valid_formats)

        # Keep only the file name, remove parent folder path
        filenames = self.filenames if keep_parent_dir else [Path(f).name for f in self.filenames]
        out_dict = {"filename": filenames}

        if isinstance(self.embs, (np.ndarray, torch.Tensor)):
            embs = self.embs.tolist()
        elif type(self.embs) == list:
            embs = self.embs
        else:
            raise TypeError("Wrong type of embeddings: {}".format(type(self.embs)))
        
        out_dict["embedding"] = embs

        if self.labels is not None:
            assert type(self.labels) == list, "[Error] Wrong type of self.labels. Must be list be found {}".format(type(self.labels))
            out_dict["label"] = self.labels


        # Pickle format
        # A more optimized way of using Pickle format than what is done here,
        # is to save each array in a different file.
        if out_path.suffix == ".pkl":
            with open(out_file, 'wb') as f:
                pickle.dump(out_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        # JSON format, heavier then Pickle but human readable
        elif out_path.suffix == ".json":
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(out_dict, f, ensure_ascii=False, indent=4)
        
        print("Embeddings saved in {}.".format(out_file))

    @abstractmethod
    def compute(self):
        """Abstract method. Child classes must override it.
        """
        pass


#----------------------------------------------------------------------------
# DINOv2 embedding

class HFDataset(Dataset):
    def __init__(self,
                 filenames: list = None,
                 from_pretrained:str = 'facebook/dinov2-base') -> None:
        self.filenames = filenames

        # Pre-processing
        self.processor = AutoImageProcessor.from_pretrained(from_pretrained)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
        img_path = self.filenames[idx]
        inputs = self.processor(images=Image.open(img_path).convert("RGB"), return_tensors="pt")
        return inputs["pixel_values"].squeeze()
    
class HierarchicalDataset(Dataset):
    def __init__(
            self,
            filenames: list,
            transform: Callable,
            device: str = "cpu",
            dtype: torch.dtype = torch.bfloat16,
        ) -> None:
        self.filenames = filenames
        self.transform = transform
        self.device = device
        self.dtype = dtype
        self._toTensor = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx: int):
        img_path = self.filenames[idx]
        img = Image.open(img_path).convert("RGB")
        img = self._toTensor(img)
        img = self.transform(img)
        return img.to(self.device, dtype=self.dtype)
        

class HFEmbedding(Embedding):
    def __init__(self,
                 filenames: str = None,
                 model_path: str = None,
                 device: str = "cpu",
                 from_pretrained:str = 'facebook/dinov2-base',
                 *args,
                 **kwargs
                 ) -> None:
        super().__init__(filenames, model_path, device, *args, **kwargs)

        # Dataset definition
        self.dataset = HFDataset(filenames=self.filenames)
        self.from_pretrained = from_pretrained

        # Load model
        if model_path is None:
            model_path = from_pretrained
        self.model = AutoModel.from_pretrained(model_path)

    def compute(self, 
            batch_size: int = 64,
            n_workers: int = 12,
            ):
        
        print("Starting embedding computation with HuggingFace model {}.".format(self.from_pretrained))
        
        # Load data from folder
        print("Loading data from input folder.")


        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=n_workers,
            pin_memory=False,
            shuffle=False, # Must be disable to match the filenames
        )

        # Load model
        print("Loading model.")
        
        self.model.to(self.device)
        self.model.eval()

        # Compute embeddings
        print("Computing embeddings.")

        all_embeddings = []
        with torch.no_grad():
            for images in tqdm(dataloader, unit="batch", desc="Predicting embeddings"):
                if self.device is not None:
                    images = images.to(self.device)
                all_embeddings.append(self.model(pixel_values=images).pooler_output.squeeze().detach().cpu())

        all_embeddings = torch.cat(all_embeddings).numpy()
        print("Embeddings: {}".format(all_embeddings.shape))

        self.embs = list(all_embeddings)

        # Filenames
        self.filenames = self.dataset.filenames

class HierarchicalEmbedding(Embedding):
    def __init__(self, model_path : str, filenames: list = None, device: str = "cpu", from_pretrained : Any=None, *args, **kwargs) -> None:
        super().__init__(filenames, device, *args, **kwargs)

        # Load model
        self.dtype = torch.bfloat16
        self.device = device
        self.model = model_from_state_file(model_path, self.device, dtype=self.dtype)
        self.model.eval()
        self.model.classifier.return_embeddings = True
        self.transform = lambda x : self.model.default_transform(x * 255.0)

    def compute(self,
                batch_size: int = 64,
                n_workers: int = 12,
                ):
        print("Starting embedding computation with Hierarchical model.")

        # Load data from folder
        print("Loading data from input folder.")

        dataset = HierarchicalDataset(filenames=self.filenames, transform=self.transform, device=torch.device("cpu"), dtype=self.dtype)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=n_workers,
            pin_memory=False,
            shuffle=False, # Must be disable to match the filenames
        )

        # Compute embeddings
        print("Computing embeddings.")

        all_embeddings = []
        with torch.no_grad():
            for images in tqdm(dataloader, unit="batch", desc="Predicting embeddings"):
                all_embeddings.append(self.model(images.to(self.device))[1].detach().cpu())

        self.embs = torch.cat(all_embeddings)
        print("Embeddings: {}".format(self.embs.shape))

        # Filenames
        self.filenames = dataset.filenames


#----------------------------------------------------------------------------
# Clustering base class

def computeAccuracy(labels, predictions):
    
    predDic = {}
    for idx in range(len(predictions)):
        prediction = predictions[idx]
        if prediction in predDic:
            predDic[prediction].append(labels[idx])
        else:
            predDic[prediction] = []
            predDic[prediction].append(labels[idx])

    TruePositives = 0
    for i, key in enumerate(predDic):    
        TruePositives += sum(predDic[key]==mode(predDic[key]))
        
    accuracy = TruePositives/len(predictions)
    return accuracy, TruePositives

def get_crop_features(crop_path: str, meta_folder : str, meta_filenames: list, time_format: Optional[str]=None):
    """Get crop date and location in the original non-cropped image.

    Parameters
    ----------
    crop_path : str
        Image crop path.
    meta_folder : str
        Path to the metadata folder.
    meta_filenames : list
        List of path of the metadata files in the metadata folder. Must be json files.
    
    Returns
    -------
    image_number : int
        Image number.
    image_date : list
        Image date.
    center : list
        Location of the center of the crop in the image: [x,y].
    """
    crop_path_split = Path(crop_path).name.split("_")

    # Default values
    image_date = datetime.strptime("20240101000000", "%Y%m%d%H%M%S")
    image_number = 0
    
    if not time_format is None:
        time_stamp = get_timestamp(crop_path, time_format)
        image_date = datetime.strptime(time_stamp, STANDARD_TIME_FORMAT)

    if "IMAGENAME" in crop_path_split:
        image_name = crop_path_split[crop_path_split.index("IMAGENAME")+1]
    # Format: data-X-number
    elif "crop" in crop_path_split:
        image_name = crop_path_split[crop_path_split.index("crop")+1]
    else:
        raise RuntimeError("Crop path {} not properly formated.'IMAGENAME_' or 'crop_ must be present.".format(crop_path))

    if not "CROPNUMBER" in crop_path_split:
        raise RuntimeError("Crop path {} not properly formated.'CROPNUMBER' must be present.".format(crop_path))
    crop_number = int(crop_path_split[crop_path_split.index("CROPNUMBER")+1])

    # Find the appropriate metadata file
    crop_meta_file = [file for file in meta_filenames if image_name in file]
    if len(crop_meta_file) == 0:
        print("Metadata not found for crop {}. Center will not be included.".format(crop_path))
        # return [image_number], image_date, [0, 0]
        return {"number": image_number, "date": image_date, "center": [0., 0]}
        # return [[0, 0]]
    else:
        crop_meta_file = crop_meta_file[0]

    # Get the bounding boxes from metadata file
    with open(os.path.join(meta_folder,crop_meta_file)) as f:
        crop_meta_dict = json.load(f)

    crop_bbox = crop_meta_dict["boxes"][crop_number]

    # Compute the center of the bounding box
    if len(crop_bbox) != 4:
        raise RuntimeError("Wrong bounding box dimensions: {}".format(crop_bbox))
    else:
        x1, y1, x2, y2 = crop_bbox
        center = [x1 + (x2-x1)/2, y1 + (y2-y1)/2]
    return {"number": image_number, "date": image_date, "center": center}

def get_crop_features_from_boxes(crop_path: str, crop_bbox: list, time_format: Optional[str]=None):
    # Default values
    image_date = datetime.strptime("20240101000000", "%Y%m%d%H%M%S")
    image_number = 0
    
    if not time_format is None:
        time_stamp = get_timestamp(crop_path, time_format)
        image_date = datetime.strptime(time_stamp, STANDARD_TIME_FORMAT)

    x1, y1, x2, y2 = crop_bbox
    center = [x1 + (x2-x1)/2, y1 + (y2-y1)/2]

    return {"number": image_number, "date": image_date, "center": center}

class Clustering:
    """
    Parameters
    ----------
    embs_file : str
        Path of the file where the embeddings are stored.
    meta_folder : str, default=None
        (Optional) Path of the metadata folder of the images. Useful for feature extraction.
    labels_file : str, default=None
        (Optional) Number of classes in the label set. Required for evaluation.
    add_features : bool, default=False
        (Optional) Whether to concatenate the metadata features to the embedding vector. NOT RECOMMENDED.
    """
    def __init__(self,
                 embs_file: str,
                 meta_folder: str = None,
                 boxes: list = None,
                 num_classes: int = None,
                 device: str = "cpu",
                 time_format: str = "%Y-%m-%d_%H-%M-%S",
                 do_dim_reduc: bool = False,
                 add_features: bool = False) -> None:
        # Path of the file where the embeddings are stored
        self.embs_file = embs_file

        # Number of classes in the evalutation set
        self.num_classes = num_classes

        # Device to run the cosine similarity matrix computation
        self.device = device

        # Store the embeddings and eventually the labels
        self.embs = None
        self.labels = None

        # Store the filenames, for reproduction and saving purpose
        self.filenames = None

        # Store the time format
        self.time_format = time_format

        # Load the embeddings
        self.load_embs()

        # Reduce the dimensionality of the embeddings
        # Intended goal: give more importance of other features
        if do_dim_reduc:
            self.dim_reduc()

        # Clustering results
        self.clusters = None

        # Metadata folder
        self.meta_folder = meta_folder
        self.meta_filenames = None
        self.features = None

        # OR hard-passed metadata
        if boxes is not None and len(boxes) != len(self.filenames):
            raise ValueError(f"Boxes and filenames must have the same length. Found {len(boxes)} boxes and {len(self.filenames)} filenames.")
        self.boxes = boxes

        # Load features from metadata
        if self.meta_folder is not None and os.path.exists(self.meta_folder):
            print("Found metadata folder. Attempting to load crops' features from metadata.")
            self.meta_filenames = [f for f in os.listdir(meta_folder) if Path(f).suffix.lower()==".json"]
            self.load_features()
        if add_features:
            self.add_features()
        if self.filenames is not None and self.boxes is not None:
            print("Load metadata from bounding boxes' list.")
            self.load_features_from_boxes()

    def load_embs(self) -> None:
        """Load the embeddings and eventually the labels.
        """
        path = Path(self.embs_file)
        valid_formats = [".json", ".pkl"]
        assert path.suffix.lower() in valid_formats, "[Error] Output file does not have a valid format: {}".format(valid_formats)

        if path.suffix == ".pkl":
            with open(self.embs_file, 'rb') as f:
                embs_dict = pickle.load(f)
        elif path.suffix == ".json":
            with open(self.embs_file) as f:
                embs_dict = json.load(f)
        else:
            raise TypeError("Wrong file format: {}".format(path.suffix))
        
        print("Opened embedding file {}. Avaliable keys {}.".format(self.embs_file, embs_dict.keys()))

        self.embs = np.array(embs_dict.get("embedding"))

        if "label" in embs_dict.keys():
            self.labels = embs_dict.get("label")

        if "filename" in embs_dict.keys():
            self.filenames = embs_dict.get("filename")
        else:
            print("Filenames not found in the emdedding file. This may cause issue during cluster saving.")

    def load_features(self) -> None:
        self.features = [get_crop_features(
                crop_path=f,
                meta_folder=self.meta_folder,
                meta_filenames=self.meta_filenames,
                time_format=self.time_format
            ) for f in tqdm(self.filenames)]
    
    def load_features_from_boxes(self):
        self.features = [get_crop_features_from_boxes(
            crop_path=f, crop_bbox=b, time_format=self.time_format) for (f,b) in zip(self.filenames, self.boxes)]
    
    def add_features(self) -> None:
        """Add features to the embeddings.
        """
        emb_n, emb_dim = self.embs.shape
        
        # Concatenate the features dimensions
        emb_dim += sum([len(l) for l in self.features[0]])

        concat_embs = np.empty((emb_n, emb_dim))
        for i, e in enumerate(self.embs):
            # Concatenate the features in an array
            feats = np.array([x for xs in self.features[i] for x in xs], dtype=float)
            concat_embs[i] = np.append(e, feats)
        self.embs = concat_embs

    def save(self, 
             out_file: str, 
             save_embs: bool = False,
             save_labels: bool = False,
             ) -> None:
        """Save the clusters.

        Currently accepted file format: csv, pkl and json.

        Parameters
        ----------
        out_file : str
            Path of the output file where to store the clusters.
            File extension must be `.csv, ``.pkl` or `json`.
        save_embs : bool, default=False
            Whether to store the embeddings along with the rest. May augment drastically the size of the final file.
        save_labels : bool, default=False
            Whether to store the labels. Required the labels to be defined.
        """
        out_path = Path(out_file)
        valid_formats = [".json", ".pkl", ".csv"]
        assert out_path.suffix.lower() in valid_formats, "[Error] Output file does not have a valid format: {}".format(valid_formats)

        # Store filenames
        if self.filenames is None:
            print("Filenames not found while saving.")
        out_dict = {"filename": self.filenames}

        # Optionally store the embdeddings
        if save_embs:
            if type(self.embs) == np.ndarray:
                embs = self.embs.tolist()
            elif type(self.embs) == list:
                embs = self.embs
            else:
                raise TypeError("Wrong type of embeddings: {}".format(type(self.embs)))
            
            out_dict["embedding"] = embs
        
        # Save the clusters
        if type(self.clusters) == np.ndarray:
            self.clusters = self.clusters.tolist()
        if self.clusters is None:
            print("No cluster found while saving.")
        else:
            out_dict["cluster"] = self.clusters 

        # Optionally store the labels
        if save_labels and self.labels is not None:
            out_dict["label"] = self.labels 

        # Pickle format
        # A more optimized way of using Pickle format than what is done here,
        # is to save each array in a different file.
        if out_path.suffix == ".pkl":
            with open(out_file, 'wb') as f:
                pickle.dump(out_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # JSON format, heavier then Pickle but human readable
        elif out_path.suffix == ".json":
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(out_dict, f, ensure_ascii=False, indent=4)

        # CSV format
        elif out_path.suffix == ".csv":
            print("Saving cluster in CSV format, the embedding will be ignored.")
            with open(out_file, 'w', newline='') as f_output:
                csv_output = csv.writer(f_output)

                # Remove None values in dict
                n_out_dict = {k:v for k,v in out_dict.items() if v is not None and k != "embedding"}

                csv_output.writerow(n_out_dict.keys())
                csv_output.writerows([*zip(*n_out_dict.values())])

        print("Clusters saved in {}.".format(out_file))

    @abstractmethod
    def compute(self):
        """Abstract method. Child classes must override it.
        """
        pass

    def compute_chunk(self, chunk_size:int = 10000, **kwargs):
        assert type(chunk_size)==int, f"Chunk size of wrong type {type(chunk_size)}"
        assert chunk_size > 0, f"Chunk size too small {chunk_size}"

        self.embs_all = self.embs.copy()
        self.features_all = self.features.copy()
        self.clusters_all = np.zeros(len(self.embs), dtype=np.int32)

        max_cluster_idx = 0
        for i in range(0,len(self.embs_all), chunk_size):
            self.embs = self.embs_all[i:i+chunk_size]
            self.features = self.features_all[i:i+chunk_size]
            self.compute(**kwargs)
            self.clusters_all[i:i+chunk_size] = self.clusters + max_cluster_idx
            max_cluster_idx = self.clusters_all.max() + 1

        self.embs = self.embs_all
        self.features = self.features_all
        self.clusters = self.clusters_all
    
def load_clusters(clusters_file: str):
    """Load the cluster file.

    Parameters
    ----------
    clusters_file : str
        File generated with cluster.Clustering.save method. Can be a .csv.

    Returns
    -------
    clusters_dict : dict
        Dictionary of the input file.
    """
    out_path = Path(clusters_file)
    valid_formats = [".csv"]
    assert out_path.suffix.lower() in valid_formats, "[Error] Cluster file does not have a valid format: {}".format(valid_formats)

    # Open CSV file
    with open(clusters_file, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        header = next(reader)
        clusters_dict = {k:[] for k in header}
        for row in reader:
            for i in range(len(row)):
                # Check if element is a digit (can be negative digit)
                clusters_dict[header[i]] += [int(row[i]) if row[i].lstrip('-').isdigit() else row[i]]
    return clusters_dict

def copy_files(out_folder, files):
    out_outliers_filenames = [os.path.join(out_folder, os.path.basename(f)) for f in files]
    with ThreadPoolExecutor(100) as exe:
        _ = [exe.submit(shutil.copyfile, path, dest) for path, dest in zip(files, out_outliers_filenames)]

def deduplicate(clusters_file: str, out_folder: str):
    """Remove image duplicates from input folder and store unique images inside out_folder.
    Non-clustered occurence are stored in an 'not-clustered' subfolder.

    Parameters
    ----------
    clusters_file : str
        File generated with cluster.Clustering.save method. Can be a .csv, .pkl or .json.
    out_folder : str
        Folder where the clustered images will be stored.
    """

    print("Copying files to output folders.")
    clusters_dict = load_clusters(clusters_file=clusters_file)
    clusters = clusters_dict["cluster"]
    filenames = clusters_dict["filename"]

    # Make the output root folder
    if not os.path.exists(out_folder):
        print("Output folder not found. Creating it.")
        os.makedirs(out_folder, exist_ok=True)

    # Make the outliers subfolder 
    unique_subfolder = os.path.join(out_folder, "not-clustered")
    if not os.path.exists(unique_subfolder):
        os.makedirs(unique_subfolder, exist_ok=True)

    # if an image is associated with a unique cluster number, then the cluster number is set to -1
    # this is to save these specific images in the root folder, instead of a subfolder
    unique_clusters, inverse_unique_clusters, count_unique_clusters = np.unique(clusters, return_counts=True, return_inverse=True)
    unique_clusters[count_unique_clusters == 1] = -1
    clusters = unique_clusters[inverse_unique_clusters]

    # Copy outliers to output folder
    outliers_filenames = np.array(filenames)[clusters == -1]
    copy_files(unique_subfolder, outliers_filenames)

    # Only keep one filename per cluster
    unique_clusters, index_unique_clusters = np.unique(clusters, return_index=True)
    ## Remove -1 clusters
    index_unique_clusters = index_unique_clusters[unique_clusters != -1]
    unique_filenames = np.array(filenames)[index_unique_clusters]
    copy_files(out_folder, unique_filenames)

def gen_cluster_subfolders(clusters_file: str, out_folder: str):
    """Generate a folder with clustered sub-folders using the output cluster file.

    Outlier (unique) images are stored directly in the output folder.

    Parameters
    ----------
    clusters_file : str
        File generated with cluster.Clustering.save method. Can be a .csv, .pkl or .json.
    out_folder : str
        Folder where the clustered images will be stored.
    """
    print("Copying files to output folders.")
    clusters_dict = load_clusters(clusters_file=clusters_file)
    clusters = clusters_dict["cluster"]
    filenames = clusters_dict["filename"]

    # Make the output root folder
    if not os.path.exists(out_folder):
        print("Output folder not found. Creating it.")
        os.makedirs(out_folder, exist_ok=True)

    # if an image is associated with a unique cluster number, then the cluster number is set to -1
    # this is to save these specific images in the root folder, instead of a subfolder
    unique_clusters, inverse_unique_clusters, count_unique_clusters = np.unique(clusters, return_counts=True, return_inverse=True)
    unique_clusters[count_unique_clusters == 1] = -1
    clusters = unique_clusters[inverse_unique_clusters].tolist()

    # Browse through the cluster dict to copy the images from input folder to subfolders
    out_filenames = []
    for i, name in enumerate(filenames):
        name = os.path.basename(name)
        cluster_folder = str(clusters[i]) if clusters[i] != -1 else "not-clustered"
        out_subfolder = os.path.join(out_folder, cluster_folder)
        if not os.path.exists(out_subfolder):
            os.makedirs(out_subfolder, exist_ok=True)
        out_filenames.append(os.path.join(out_subfolder, name))

    # copyfile = lambda x: shutil.copyfile(x[0], x[1])
    # with ThreadPool(32) as p:
    #     p.map(copyfile, zip(abs_filenames, out_filenames))
    # for path, dest in zip(abs_filenames, out_filenames):
    #     shutil.copyfile(path, dest)
    with ThreadPoolExecutor(100) as exe:
        _ = [exe.submit(shutil.copyfile, path, dest) for path, dest in zip(filenames, out_filenames)]


#----------------------------------------------------------------------------
# Clustering method

# Transitive closure functions copied from https://github.com/darsa-group/flat-bug/blob/dev_experiments/src/flat_bug/nms.py
@torch.jit.script
def _compute_transitive_closure_cpu(adjacency_matrix : torch.Tensor) -> torch.Tensor:
    """
    Computes the transitive closure of a boolean matrix.
    """
    csize = adjacency_matrix.shape[0]
    # Check for possible overflow
    if csize > 2**31 - 1:
        raise ValueError(f"Matrix is too large ({csize}x{csize}) for CPU computation")
    # We convert to torch.int16 to avoid overflow when squaring the matrix and ensure torch compatibility
    closure = adjacency_matrix.to(torch.int32) 
    # Expand the adjacency matrix to the transitive closure matrix, by squaring the matrix and clamping the values to 1 - each step essentially corresponds to one step of parallel breadth-first search for all nodes
    last_max = torch.zeros(csize, dtype=torch.int32)
    for _ in range(int(torch.log2(torch.tensor(csize, dtype=torch.float32)).ceil())):
        this_square = torch.matmul(closure, closure)
        this_max = this_square.max(dim=0).values
        if (this_max == last_max).all():
            break
        closure[:] = this_square.clamp(max=1) # We don't need to worry about overflow, since overflow results in +inf, which is clamped to 1
        last_max = this_max
    # Convert the matrix back to boolean and return it
    return closure > 0.5

@torch.jit.script
def _compute_transitive_closure_cuda(adjacency_matrix : torch.Tensor) -> torch.Tensor:
    """
    Computes the transitive closure of a boolean matrix.
    """
    # torch._int_mm only supports matrices such that the output is larger than 32x32 and a multiple of 8
    if len(adjacency_matrix) < 32:
        padding = 32 - len(adjacency_matrix)
    elif len(adjacency_matrix) % 8 != 0:
        padding = 8 - len(adjacency_matrix) % 8
    else:
        padding = 0
    # Convert the adjacency matrix to float16, this is just done to ensure that the values don't overflow when squaring the matrix before clamping - if there existed a "or-matrix multiplication" for boolean matrices, this would not be necessary
    closure = torch.nn.functional.pad(adjacency_matrix, (0, padding, 0, padding), value=0.).to(torch.int8) 
    # Expand the adjacency matrix to the transitive closure matrix, by squaring the matrix and clamping the values to 1 - each step essentially corresponds to one step of parallel breadth-first search for all nodes
    last_max = torch.zeros(len(closure), dtype=torch.int32, device=closure.device)
    for _ in range(int(torch.log2(torch.tensor(adjacency_matrix.shape[0], dtype=torch.float16)).ceil())):
        this_square = torch._int_mm(closure, closure)
        this_max = this_square.max(dim=0).values
        if (this_max == last_max).all():
            break
        closure[:] = this_square >= 1
        last_max = this_max
    # Convert the matrix back to boolean and remove the padding
    closure = (closure > 0.5)
    if padding > 0:
        closure = closure[:-padding, :-padding]
    return closure

# @torch.jit.script
def compute_transitive_closure(adjacency_matrix : torch.Tensor) -> torch.Tensor:
    if len(adjacency_matrix.shape) != 2 or adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError(f"Matrix must be of shape (n, n), not {adjacency_matrix.shape}")
    # If the matrix is a 0x0, 1x1 or 2x2 matrix, the transitive closure is the matrix itself, since there are no transitive relations
    if len(adjacency_matrix) <= 2:
        return adjacency_matrix    
    # There can be a quite significant difference in performance between the CPU and GPU implementation, however this function is not the bottleneck, so it might not be noticeable in practice
    if adjacency_matrix.is_cuda:
        print("CUDA")
        return _compute_transitive_closure_cuda(adjacency_matrix)
    else:
        print("CPU")
        return _compute_transitive_closure_cpu(adjacency_matrix)

# Cosine similarity function copied from https://github.com/idealo/imagededup/blob/master/imagededup/methods/cnn.py
def parallelise(function: Callable, data: List, verbose: bool, num_workers: int) -> List:
    num_workers = 1 if num_workers < 1 else num_workers  # Pool needs to have at least 1 worker.
    pool = Pool(processes=num_workers)
    results = list(
        tqdm(pool.imap(function, data, 100), total=len(data), disable=not verbose)
    )
    pool.close()
    pool.join()
    return results

def cosine_similarity_chunk(t: Tuple) -> np.ndarray:
    return cosine_similarity(t[0][t[1][0]: t[1][1]], t[0]).astype('float16')

def get_cosine_similarity(
    X: np.ndarray,
    verbose: bool = True,
    chunk_size: int = 1000,
    threshold: int = 10000,
    num_workers: int = 0,
) -> np.ndarray:
    n_rows = X.shape[0]

    if n_rows <= threshold:
        return cosine_similarity(X)

    else:
        print(
            'Large feature matrix thus calculating cosine similarities in chunks...'
        )
        start_idxs = list(range(0, n_rows, chunk_size))
        end_idxs = start_idxs[1:] + [n_rows]

        if num_workers > 0:
            cos_sim = parallelise(
                cosine_similarity_chunk,
                [(X, idxs) for i, idxs in enumerate(zip(start_idxs, end_idxs))],
                verbose,
                num_workers,
            )
        else:
            cos_sim = tuple(
                cosine_similarity_chunk((X, idxs))
                for idxs in tqdm(zip(start_idxs, end_idxs), total=len(start_idxs))
            )
        return np.vstack(cos_sim)

class CosineClustering(Clustering):
    """Good and fast.
    """
    def compute(self, threshold: float = 0.9):
        print("Starting clustering with cosine similarity algorithm with threshold {}".format(threshold))
        print("Computing cosine similarity matrix.")
        # cosine = get_cosine_similarity(self.embs, num_workers=cpu_count())

        embs = torch.from_numpy(self.embs).to(self.device)
        embs /= embs.norm(2, 1, keepdim=True)
        cosine = embs @ embs.T

        if self.features is not None:
            # Center distance matrix
            centers = [f["center"] for f in self.features]
            centers_dist = pdist(centers, "euclidean")

            # Apply a distance threshold, TODO: by default = 1/10 of the max distance OR = 400 ??
            dist_th = 100
            centers_dist = squareform(centers_dist)
            S = (centers_dist < dist_th)
            S = torch.from_numpy(S).to(self.device)

            # Time distance matrix
            times = [f["date"] for f in self.features]
            times = [(t-min(times)).total_seconds() for t in times]
            times_dist = pdist(np.expand_dims(times,-1), "euclidean")

            # Apply a time threshold, TODO: default=10secondes
            time_th_large = 300
            time_th_small = 30
            times_dist = squareform(times_dist)
            TL = (times_dist < time_th_large)
            TL = torch.from_numpy(TL).to(self.device)
            TS = (times_dist < time_th_small)
            TS = torch.from_numpy(TS).to(self.device)

            # cosine = centers_kernel * times_kernel * cosine
            # del centers_kernel, times_kernel

        E = cosine >= threshold
        is_connected = E
        if self.features is not None:
            is_connected &= (S & TL) | (TS & E)
        del cosine

        print("Computing transitive closure")
        self.clusters = np.zeros(len(self.embs), dtype=np.int32)
        is_connected = compute_transitive_closure(is_connected)

        print("Computing clusters.")
        for node in tqdm(range(len(self.clusters))):
            if self.clusters[node]:
                continue
            self.clusters[is_connected[node].cpu().numpy()] = node

#----------------------------------------------------------------------------
# Runners

VALID_NAMES_EMBED = {
    "dino": HFEmbedding,
    "hierarchical": HierarchicalEmbedding,
}

def run_embed(
    method: Callable,
    out_folder: str,
    model_path: str,
    filenames: list = None,
    from_pretrained: str = None,
    device: str = "cpu",
    time_format: str = "%YYYY-%mm-%dd_%HH-%MM-%SS"
    ) -> str:
    """Run the embedding workflow.
    """

    if not os.path.exists(out_folder):
        print("Output folder for embeddings not found. Creating it.")
        os.makedirs(out_folder, exist_ok=True)

    # Create an unique output filename with the date and time, plus the method name
    out_file = datetime.now().strftime(f"{convert_timeformat(time_format)}_") + method.__name__ + "_embs.pkl"
    out_file = os.path.join(out_folder, out_file)

    # Compute the embeddings
    emb = method(
        filenames = filenames,
        model_path = model_path,
        device = device,
        from_pretrained = from_pretrained,
        time_format = time_format
    )
    emb.compute()
    emb.save(out_file)

    # Return the output file
    return out_file

VALID_NAMES_CLUSTER = {
    "cosine": CosineClustering,
}

def run_cluster(
    method: Callable,
    embs_file: str,
    meta_folder: str,
    out_folder: str,
    boxes: list = None,
    num_classes: int = None,
    save_embs: bool = False,
    save_labels: bool = False,
    device: str = "cpu",
    threshold: float = None,
    chunk_size: int = 10000,
    time_format: str = "%YYYY-%mm-%dd_%HH-%MM-%SS"
    ) -> str:
    """Run the clustering workflow.
    """

    if not os.path.exists(out_folder):
        print("Output folder for clustering results not found. Creating it.")
        os.makedirs(out_folder, exist_ok=True)

    # Create an unique output filename with the date and time, plus the method name
    ext = ".pkl" if save_embs else ".csv"
    out_file = datetime.now().strftime(f"{convert_timeformat(time_format)}_") + method.__name__ + "_clusters" + ext
    out_file = os.path.join(out_folder, out_file)

    # Compute the clusters, evaluate them if possible
    cluster = method(
        embs_file = embs_file,
        boxes = boxes,
        meta_folder = meta_folder,
        num_classes = num_classes,
        device = device,
        time_format = time_format
    )
    
    cluster.compute_chunk(chunk_size) if threshold is None else cluster.compute_chunk(chunk_size=chunk_size, threshold=threshold)
    if num_classes is not None and cluster.labels is not None:
        print("Found cluster labels. Evaluating.")
        cluster.eval()
    cluster.save(
        out_file, 
        save_embs = save_embs,
        save_labels = save_labels)
    
    print("Clustering done. Clusters can be found in {}.".format(out_file))

    # Return the output file
    return out_file, {"filename": cluster.filenames, "cluster": cluster.clusters}


def main(args : Dict):
    filenames = args.get("input", None)
    if filenames is None:
        raise ValueError("Input files must be specified.")
    filenames = get_images(filenames)
    boxes = args.get("boxes", None)
    # model_path = args.get("model_path", "facebook/dinov2-base")
    out_folder = args.get("out_folder", None)
    if out_folder is None:
        raise ValueError("Output folder must be specified.")
    time_format = args.get("time_format", "%YYYY-%mm-%dd_%HH-%MM-%SS")
    device = args.get("device", None)
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    embed = args.get("embed", "dino")
    from_pretrained = args.get("from_pretrained", None)
    cluster = args.get("cluster", "cosine")
    num_classes = args.get("num_classes", None)
    save_embs = args.get("save_embs", False)
    save_labels = args.get("save_labels", False)
    meta_folder = args.get("meta_folder", None)
    threshold = args.get("threshold", None)
    dedup = args.get("dedup", False)
    gen_cluster_dirs = args.get("gen_cluster_dirs", False)

    match embed:
        case "dino":
            model_path = "facebook/dinov2-base"
        case "hierarchical":
            model_path = "efficientnet_v2_s___hierarchical.state"
            if not os.path.exists(model_path):
                urlretrieve(ERDA_MODEL_ZOO_TEMPLATE.format("hierarchical/effnetv2s_sgfoc_train_v3_1/efficientnet_v2_s___epoch_8_batch_22000.state"), model_path)
        case _:
            raise ValueError("Embedding method not recognized: {}".format(embed))

    # Compute the embeddings
    embs_file = run_embed(
        method = VALID_NAMES_EMBED[embed],
        filenames = filenames,
        out_folder = out_folder,
        model_path = model_path,
        device = device,
        from_pretrained = from_pretrained,
        time_format = time_format
    )

    # Compute the clusters
    clusters_file, clusters = run_cluster(
        method = VALID_NAMES_CLUSTER[cluster],
        embs_file = embs_file,
        boxes = boxes,
        meta_folder = meta_folder,
        out_folder = out_folder,
        num_classes = num_classes,
        save_embs = save_embs,
        save_labels = save_labels,
        device = device,
        threshold = threshold,
        time_format = time_format
    )

    # If not precised, the output folder will have the same name as the input folder with an additional '_clustered' suffix.
    cluster_folder = os.path.join(out_folder, "clusters")

    # Generate the subfolders
    if dedup:
        deduplicate(
            clusters_file = clusters_file,
            out_folder = cluster_folder,
        )
    if gen_cluster_dirs:
        gen_cluster_subfolders(
            clusters_file = clusters_file,
            out_folder = cluster_folder,
        )

    print("Image crops successfully clustered in {}".format(out_folder))
    return clusters

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Embedding and clustering computation.")
    parser.add_argument('-i', '--input', type=str, nargs="+", required=True,
        help='Path(s), director(y/ies) or glob(s) to such. If it is a .txt file, then it should contain lines corresponding to the former. Outputs will be saved in the output directory.')
    parser.add_argument("-o", "--out_folder", type=str, required=True,
        help="(Optional) Output folder for the clustered images.")
    parser.add_argument('--time_format', type=str, default="%YYYY-%mm-%dd_%HH-%MM-%SS", required=False,
        help='Format for the time stamp in image file names.')
    parser.add_argument("-d", "--device", type=str,
        help="(Optional) Device used to run the embedding model. Defaults to cuda:0 if available, else 'cpu'.")
    parser.add_argument("-e", "--embed", type=str, default="dino",
        help="(Optional) Name of the embedding method. Valid names: {}".format(VALID_NAMES_EMBED.keys()))
    parser.add_argument("-fp", "--from_pretrained", type=str, default='facebook/dinov2-base',
        help="(Optional) HuggingFace model for embedding computation. Only required if 'embed=dino'.")
    parser.add_argument("-kp", "--dedup", default=False,  action='store_true', dest='dedup',
        help="(Optional) Whether to deduplicate the images. If True, then all images will be stored in subfolders.") 
    parser.add_argument("-gs", "--gen_cluster_dirs", default=False,  action='store_true', dest='gen_cluster_dirs',
        help="(Optional) Whether to generate an output folder with one subfolder for each cluster. Outliers are left aside.") 
    parser.add_argument("-c", "--cluster", type=str, default="cosine",
        help="(Optional) Name of the clustering method. Valid names: {}".format(VALID_NAMES_CLUSTER.keys()))
    parser.add_argument("-mf", "--meta_folder", type=str, default=None,
        help="(Optional) Path of the metadata folder of the images. Useful for feature extraction.")
    parser.add_argument("-x", "--num_classes", type=int, default=None,
        help="(Optional) Number of classes of the labels. Required for the evaluation") 
    parser.add_argument("-se", "--save_embs", default=False,  action='store_true', dest='save_embs',
        help="(Optional) Whether to save the embeddings in the output file.") 
    parser.add_argument("-sl", "--save_labels", default=False,  action='store_true', dest='save_labels',
        help="(Optional) Whether to save the labels in the output file. Labels must exit.") 
    parser.add_argument("-th", "--threshold", type=float, default = None,
        help="(Optional) Threshold for the cosine similarity clustering.")
    args = parser.parse_args()

    main(vars(args))