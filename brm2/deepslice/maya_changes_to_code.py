### deepslice_dataset.py >> __getitem__ - DONE
# made changes 

def __getitem__(self, index) -> DeepSliceDatasetSample:
    """Get the dataset sample at the given index."""
    if index >= len(self._labels):
        raise IndexError
    filename = self._labels.iloc[index].filename
    image = read_image(filename) # reads image into matrix of pixels 
    if self._transform is not None:
        image = self._transform(image=image)["image"]
    ap_value = torch.as_tensor([float(self._labels.iloc[index]["ap"])]) # keep this for now - I think we need it for create_slice
    # MAYA - added this: 
    ox = torch.as_tensor([float(self._labels.iloc[index]["ox"])])
    oy = torch.as_tensor([float(self._labels.iloc[index]["oy"])])
    oz = torch.as_tensor([float(self._labels.iloc[index]["oz"])])
    ux = torch.as_tensor([float(self._labels.iloc[index]["ux"])])
    uy = torch.as_tensor([float(self._labels.iloc[index]["uy"])])
    uz = torch.as_tensor([float(self._labels.iloc[index]["uz"])])
    vx = torch.as_tensor([float(self._labels.iloc[index]["vx"])])
    vy = torch.as_tensor([float(self._labels.iloc[index]["vy"])])
    vz = torch.as_tensor([float(self._labels.iloc[index]["vz"])])
    atlas_name = self._labels.iloc[index]["atlas_name"]
    return {
        "image": image,
        "ap": ap_value, 
        "ox": ox,
        "oy": oy,
        "oz": oz,
        "ux": ux,
        "uy": uy,
        "uz": uz,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "atlas_name": atlas_name,
        "filename": filename,
    }


### yml
# data>>data_sources: change train to this - DONE
"""
train: # these will be initializion params for a DataSource
  - images_path: /home/ben/data/allen-resized # absolute paths
    labels_path: DS_gt_all_labels.csv # I added this to the deepslice folder
    atlas_name: allen_mouse_25um
    type: real
"""


