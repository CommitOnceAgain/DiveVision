from abc import ABC


class AbstractDataset(ABC):
    name: str

    def load_data(self):
        raise NotImplementedError
