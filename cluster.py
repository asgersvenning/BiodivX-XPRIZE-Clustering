import os, re, glob, shutil, json, csv, zipfile, pickle, argparse

from typing import Callable, List, Tuple, Dict, Optional, Any

from abc import abstractmethod
from datetime import datetime
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

from statistics import mode
from tqdm import tqdm
from PIL import Image

import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity

ERDA_MODEL_ZOO_TEMPLATE = "https://anon.erda.au.dk/share_redirect/aRbj0NCBkf/{}"


#----------------------------------------------------------------------------
# Embedding base class

class Embedding:
    """Abstract class to manage inputs (images crops) and outputs (embeddings).

    The class method `self.compute` is abstract and must be overridden in child-classes.
    """

    def __init__(self,
                 in_folder: str,
                 model_path: str,
                 device: str = "cpu",
                 **kwargs,
                 ) -> None:

        # Input folder of image crops
        self.in_folder = in_folder

        self.model_path = model_path

        # Device on which computations will be done
        self.device = device

        # Array of embeddings
        self.embs = None

        # Labels of the embeddings
        # Can stay None if no labels exist
        self.labels = None

        # List of filenames of the images
        self.filenames = None 

    def save(self, out_file : str) -> None: # TODO: allows to save absolute path of images instead of relative.
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
        if self.filenames is not None:
            filenames = [Path(f).name for f in self.filenames]
        out_dict = {"filename": filenames}

        if type(self.embs) == np.ndarray:
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
    def __init__(self, in_folder: str, from_pretrained:str = 'facebook/dinov2-base') -> None:
        self.in_folder = in_folder
        self.filenames = os.listdir(self.in_folder)

        # Pre-processing
        self.processor = AutoImageProcessor.from_pretrained(from_pretrained)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.in_folder, self.filenames[idx])
        inputs = self.processor(images=Image.open(img_path).convert("RGB"), return_tensors="pt")
        return inputs["pixel_values"].squeeze()

class HFEmbedding(Embedding):
    def __init__(self, in_folder: str, model_path: str = None, device: str = "cpu", from_pretrained:str = 'facebook/dinov2-base') -> None:
        super().__init__(in_folder, model_path, device)

        # Dataset definition
        self.dataset = HFDataset(in_folder=self.in_folder)
        self.from_pretrained = from_pretrained

        # Load model
        if model_path is None:
            model_path = from_pretrained
        self.model = AutoModel.from_pretrained(model_path)

    def compute(self, 
            batch_size: int = 128,
            n_workers: int = 12,
            ):
        
        print("Starting embedding computation with HuggingFace model {}.".format(self.from_pretrained))
        
        # Load data from folder
        print("Loading data from input folder.")


        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=n_workers,
            pin_memory=True,
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

def get_crop_features(crop_path: str, meta_folder : str, meta_filenames: list):
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
    image_date = datetime.strptime("202401010000", "%Y%m%d%H%M%S")
    image_number = 0
    
    # find date with regex
    pattern = r'(?<!\d)\d{14}(?!\d)'
    matches = re.findall(pattern, crop_path)
    if len(matches)==1:
        image_date = datetime.strptime(matches[0], "%Y%m%d%H%M%S")

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

class Clustering:
    """
    Parameters
    ----------
    embs_file : str
        Path of the file where the embeddings are stored.
    meta_folder : str, default=None
        (Optional) Path of the metadata folder of the images. Useful for feature extraction.
    in_folder : str, default=None
        (Optional) Path of the crop folder. Required only for metadata extraction.
    labels_file : str, default=None
        (Optional) Number of classes in the label set. Required for evaluation.
    add_features : bool, default=False
        (Optional) Whether to concatenate the metadata features to the embedding vector. NOT RECOMMENDED.
    """
    def __init__(self,
                 embs_file: str,
                 meta_folder: str = None,
                 in_folder: str = None,
                 num_classes: int = None,
                 device: str = "cpu",
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
        self.in_folder = in_folder
        self.meta_filenames = None
        self.features = None

        # Load features from metadata
        if self.filenames is not None and self.meta_folder is not None and os.path.exists(self.meta_folder):
            if self.in_folder is None:
                raise ValueError("If metadata extraction is intended, the input folder must be precised. Please set `in_folder` argument.")

            print("Found metadata folder. Attempting to load crops' features from metadata.")
            self.meta_filenames = [f for f in os.listdir(meta_folder) if Path(f).suffix.lower()==".json"]
            self.load_features()
        if add_features:
            self.add_features()

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
                crop_path=os.path.join(self.in_folder, f),
                meta_folder=self.meta_folder,
                meta_filenames=self.meta_filenames
            ) for f in tqdm(self.filenames)]
    
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

def copy_files(in_folder, out_folder, files):
    in_outliers_filenames = [os.path.join(in_folder, f) for f in files]
    out_outliers_filenames = [os.path.join(out_folder, f) for f in files]
    with ThreadPoolExecutor(100) as exe:
        _ = [exe.submit(shutil.copyfile, path, dest) for path, dest in zip(in_outliers_filenames, out_outliers_filenames)]

def deduplicate(in_folder: str, clusters_file: str, out_folder: str):
    """Remove image duplicates from input folder and store unique images inside out_folder.
    Non-clustered occurence are stored in an 'not-clustered' subfolder.

    Parameters
    ----------
    in_folder : str
        Input image folder. 
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
    copy_files(in_folder, unique_subfolder, outliers_filenames)

    # Only keep one filename per cluster
    unique_clusters, index_unique_clusters = np.unique(clusters, return_index=True)
    ## Remove -1 clusters
    index_unique_clusters = index_unique_clusters[unique_clusters != -1]
    unique_filenames = np.array(filenames)[index_unique_clusters]
    copy_files(in_folder, out_folder, unique_filenames)

def gen_cluster_subfolders(in_folder: str, clusters_file: str, out_folder: str):
    """Generate a folder with clustered sub-folders using the output cluster file.

    Outlier (unique) images are stored directly in the output folder.

    Parameters
    ----------
    in_folder : str
        Input image folder. 
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
    for i in range(len(filenames)):
        if clusters[i] == -1:
            out_filenames.append(os.path.join(out_folder, filenames[i]))
        # Create a subfolder with the cluster number and copy image inside
        else:
            out_subfolder = os.path.join(out_folder,str(clusters[i]))
            if not os.path.exists(out_subfolder):
                os.makedirs(out_subfolder, exist_ok=True)
            out_filenames.append(os.path.join(out_subfolder, filenames[i]))

    abs_filenames = [os.path.join(in_folder, f) for f in filenames]

    # copyfile = lambda x: shutil.copyfile(x[0], x[1])
    # with ThreadPool(32) as p:
    #     p.map(copyfile, zip(abs_filenames, out_filenames))
    # for path, dest in zip(abs_filenames, out_filenames):
    #     shutil.copyfile(path, dest)
    with ThreadPoolExecutor(100) as exe:
        _ = [exe.submit(shutil.copyfile, path, dest) for path, dest in zip(abs_filenames, out_filenames)]


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
    def compute(self, threshold: float = 0.8):
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
            dist_th = 400
            centers_dist = squareform(centers_dist)
            centers_kernel = (centers_dist < dist_th)
            centers_kernel = torch.from_numpy(centers_kernel).to(self.device)

            # Time distance matrix
            times = [f["date"] for f in self.features]
            times = [(t-min(times)).total_seconds() for t in times]
            times_dist = pdist(np.expand_dims(times,-1), "euclidean")

            # Apply a time threshold, TODO: default=10secondes
            time_th = 180
            times_dist = squareform(times_dist)
            times_kernel = (times_dist < time_th)
            times_kernel = torch.from_numpy(times_kernel).to(self.device)

            cosine = centers_kernel * times_kernel * cosine
            del centers_kernel, times_kernel

        is_connected = (cosine >= threshold) 
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
}

def run_embed(
    method: Callable,
    in_folder: str,
    out_folder: str,
    model_path: str,
    from_pretrained: str = None,
    device: str = "cpu",
    ) -> str:
    """Run the embedding workflow.
    """

    if not os.path.exists(out_folder):
        print("Output folder for embeddings not found. Creating it.")
        os.makedirs(out_folder, exist_ok=True)

    # Create an unique output filename with the date and time, plus the method name
    out_file = datetime.now().strftime("%Y%m%d-%H%M%S_") + method.__name__ + "_embs.pkl"
    out_file = os.path.join(out_folder, out_file)

    # Compute the embeddings
    emb = method(
        in_folder = in_folder,
        model_path = model_path,
        device = device,
        from_pretrained = from_pretrained,
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
    in_folder: str, 
    meta_folder: str,
    out_folder: str,
    num_classes: int = None,
    save_embs: bool = False,
    save_labels: bool = False,
    device: str = "cpu",
    threshold: float = None,
    ) -> str:
    """Run the clustering workflow.
    """

    if not os.path.exists(out_folder):
        print("Output folder for clustering results not found. Creating it.")
        os.makedirs(out_folder, exist_ok=True)

    # Create an unique output filename with the date and time, plus the method name
    ext = ".pkl" if save_embs else ".csv"
    out_file = datetime.now().strftime("%Y%m%d-%H%M%S_") + method.__name__ + "_clusters" + ext
    out_file = os.path.join(out_folder, out_file)

    # Compute the clusters, evaluate them if possible
    cluster = method(
        embs_file = embs_file,
        in_folder = in_folder,
        meta_folder = meta_folder,
        num_classes = num_classes,
        device = device)
    
    cluster.compute() if threshold is None else cluster.compute(threshold)
    if num_classes is not None and cluster.labels is not None:
        print("Found cluster labels. Evaluating.")
        cluster.eval()
    cluster.save(
        out_file, 
        save_embs = save_embs,
        save_labels = save_labels)
    
    print("Clustering done. Clusters can be found in {}.".format(out_file))

    # Return the output file
    return out_file


def main(args : Dict):
    in_folder = args["in_folder"]
    model_path = args["model_path"]
    out_folder = args.get("out_folder", None)
    device = args.get("device", "cpu")
    embed = args.get("embed", "dino")
    from_pretrained = args.get("from_pretrained", None)
    cluster = args.get("cluster", "cosine")
    num_classes = args.get("num_classes", None)
    save_embs = args.get("save_embs", False)
    save_labels = args.get("save_labels", False)
    meta_folder = args.get("meta_folder", None)
    intermediate_folder = args.get("inter_folder", "results/")
    threshold = args.get("threshold", None)
    dedup = args.get("dedup", False)


    # If in_folder is a zip, then unzip it
    if os.path.isdir(in_folder):
        pass
    elif in_folder.endswith(".zip"):
        in_folder = Path(in_folder)
        ext = in_folder.suffix
        if ext == ".zip":
            # Create unzipping folder
            unzip_folder = os.path.join(in_folder.parent, in_folder.stem)

            print("Found zip file as input. Unzipping it in {}.".format(unzip_folder))
            with zipfile.ZipFile(in_folder, 'r') as zip_ref:
                zip_ref.extractall(unzip_folder)

            # Check if the zip file did not contain another folder
            in_subfolder = list(set([os.path.dirname(p) for p in glob.glob(os.path.join(unzip_folder,"*/*"))]))
            if len(in_subfolder) == 1:
                print("Found one subfolder in zip file {}. Will use it as image folder".format(in_subfolder))
                in_folder = in_subfolder[0]
            elif len(in_subfolder) > 1:
                print("Too many subfolders in zip file: {}".format(in_subfolder))
                return
            else:
                in_folder = unzip_folder
        else:
            print("Wrong zip format. Found {} instead of zip.".format(ext))
            raise TypeError	
    else:
        raise FileNotFoundError("Input folder not found: {}".format(in_folder))

    # Compute the embeddings
    embs_file = run_embed(
        method = VALID_NAMES_EMBED[embed],
        in_folder = in_folder,
        out_folder = intermediate_folder,
        model_path = model_path,
        device = device,
        from_pretrained = from_pretrained,
    )

    # Compute the clusters
    clusters_file = run_cluster(
        method = VALID_NAMES_CLUSTER[cluster],
        embs_file = embs_file,
        in_folder = in_folder,
        meta_folder = meta_folder,
        out_folder = intermediate_folder,
        num_classes = num_classes,
        save_embs = save_embs,
        save_labels = save_labels,
        device = device,
        threshold = threshold
    )

    # If not precised, the output folder will have the same name as the input folder with an additional '_clustered' suffix.
    if out_folder is None:
        out_folder = str(Path(in_folder)) + "_clustered"

    # Generate the subfolders
    out_function = deduplicate if dedup else gen_cluster_subfolders
    out_function(
        in_folder = in_folder,
        clusters_file = clusters_file,
        out_folder = out_folder,
    )

    print("Image crops successfully clustered in {}".format(out_folder))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Embedding and clustering computation.")
    parser.add_argument("-i", "--in_folder", type=str, 
        help="Input folder with the image crops.")
    parser.add_argument("-m", "--model_path", type=str, default=None,
        help="(Optional) Model path for the embedding. If not found locally, a default model will be downloaded.") 
    parser.add_argument("-o", "--out_folder", type=str, default=None,
        help="(Optional) Output folder for the clustered images.")
    parser.add_argument("-d", "--device", type=str, default="cpu",
        help="(Optional) Device used to run the embedding model.")
    parser.add_argument("-e", "--embed", type=str, default="dino",
        help="(Optional) Name of the embedding method. Valid names: {}".format(VALID_NAMES_EMBED.keys()))
    parser.add_argument("-fp", "--from_pretrained", type=str, default='facebook/dinov2-base',
        help="(Optional) HuggingFace model for embedding computation. Only required if 'embed=dino'.")
    parser.add_argument("-kp", "--dedup", default=False,  action='store_true', dest='dedup',
        help="(Optional) Whether to deduplicate the images. If True, then all images will be stored in subfolders.") 
    parser.add_argument("-c", "--cluster", type=str, default="cosine",
        help="(Optional) Name of the clustering method. Valid names: {}".format(VALID_NAMES_CLUSTER.keys()))
    parser.add_argument("-mf", "--meta_folder", type=str, default=None,
        help="(Optional) Path of the metadata folder of the images. Useful for feature extraction.")
    parser.add_argument("-x", "--num_classes", type=int, default=None,
        help="(Optional) Number of classes of the labels. Required for the evaluation") 
    parser.add_argument("-if", "--inter_folder", type=str, default="results/",
        help="(Optional) Intermediate folder to store embeddings and clusters data. Default='./results/'")
    parser.add_argument("-se", "--save_embs", default=False,  action='store_true', dest='save_embs',
        help="(Optional) Whether to save the embeddings in the output file.") 
    parser.add_argument("-sl", "--save_labels", default=False,  action='store_true', dest='save_labels',
        help="(Optional) Whether to save the labels in the output file. Labels must exit.") 
    parser.add_argument("-th", "--threshold", type=float, default = None,
        help="(Optional) Threshold for the cosine similarity clustering.")
    args = parser.parse_args()

    main(vars(args))