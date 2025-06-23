
class Dataset:
    def __init__(self, data) -> None:
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __iter__(self):
        for item in self.data:
            yield item

    def __repr__(self):
        return f"Dataset({len(self)})"