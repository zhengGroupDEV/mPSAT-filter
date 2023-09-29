from tap import Tap


class DsGeneratorArgParser(Tap):
    in_data: str
    out_dir: str
    mpid_type_hash: str = "data/mpid_type.json"
    data_format: str = "feather"
    overwrite: bool = False
    no_mutate: bool = False
    no_combine: bool = False
    ensure_nomutate: bool = False
    no_split: bool = False
    ptrain: float = 0.7
    pval: float = 0.2
    ptest: float = 0.1
    max_std: float = 0.2
    max_num_per_id: int = 1000
    merge_val_test: bool = False
    save_image: bool = False
    label_hash: str = ""

    def configure(self) -> None:
        self.add_argument("-i", "--in_data")
        self.add_argument("-o", "--out_dir")
