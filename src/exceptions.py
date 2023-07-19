class InvalidCaryFormatError(Exception):
    def __init__(self):
        self.message: str = "The format of the Cary file is not valid for parsing. " \
                            "Please make sure to read the correct file."

    def __str__(self):
        return f"{self.message}"
