class InvalidCaryFormatError(Exception):
    def __init__(self):
        self.message: str = "The format of the Cary file is not valid for parsing. " \
                            "Please make sure to read the correct file."

    def __str__(self):
        return f"{self.message}"


class InvalidHyperparameterError(Exception):
    def __init__(self):
        self.message: str = "The format of the Hyperparameters does not match Key and Value." \
                            "If this needs to be implemented, reach out to mweber95."

    def __str__(self):
        return f"{self.message}"


class InvalidHyperparameterHeaderError(Exception):
    def __init__(self):
        self.message: str = "The first value of a hyperparameter section doesn't match " \
                            "the implemented options for the 'Cary'." \
                            "If this needs to be implemented, reach out to mweber95."

    def __str__(self):
        return f"{self.message}"
