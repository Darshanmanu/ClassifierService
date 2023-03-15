import os

# Directory service which will process the dataset from the Dataset directory
# and get the category and training datasets.


class DirectoryHelper:
    def __init__(self, location):
        print("Processing the directory from location ", location)
        self._location = location

    def get_training_dataset(self):
        file_dict = {}
        for filename in os.listdir(self._location):
            print("Processing the filename ", filename)
            file_path = os.path.join(self._location, filename)

            # read the dataset.
            with open(file_path, "r") as file:
                file_content = file.read()

            file_dict[filename] = file_content.split('\n')

        return file_dict
