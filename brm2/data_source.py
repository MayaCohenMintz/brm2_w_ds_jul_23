from typing import Literal, TypedDict


DataSourceType = Literal["synth", "real"]


class DataSource(TypedDict):
    images_path: str
    labels_path: str
    atlas_name: str
    type: DataSourceType
