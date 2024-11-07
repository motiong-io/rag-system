import pickle

class PickleManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self._load_data()

    def _load_data(self):
        try:
            with open(self.file_path, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return {}

    def _save_data(self):
        with open(self.file_path, 'wb') as file:
            pickle.dump(self.data, file)

    def add_entry(self, key, value):
        if key in self.data:
            print(f"Key '{key}' already exists. Use update_entry to modify.")
            return
        self.data[key] = value
        self._save_data()
        print(f"Added {key}: {value}")

    def delete_entry(self, key):
        if key in self.data:
            del self.data[key]
            self._save_data()
            print(f"Deleted {key}")
        else:
            print(f"Key '{key}' not found.")

    def update_entry(self, key, value):
        if key in self.data:
            self.data[key] = value
            self._save_data()
            print(f"Updated {key}: {value}")
        else:
            print(f"Key '{key}' not found. Use add_entry to insert new entry.")

    def retrieve_entry(self, key):
        if key in self.data:
            return self.data[key]
        else:
            print(f"Key '{key}' not found")
            return None

    def list_entries(self):
        return self.data.items()

# 示例使用
if __name__ == "__main__":
    pkl_path="data/q0_contextual_db/contextual_vector_db.pkl"
    manager = PickleManager(pkl_path)
    # manager.add_entry('key1', 'value1')
    # manager.update_entry('key1', 'new_value1')
    # print(manager.retrieve_entry('key1'))
    # manager.delete_entry('key1')
    print(dict(manager.list_entries()).keys())