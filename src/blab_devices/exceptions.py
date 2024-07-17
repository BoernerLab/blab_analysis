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


class NoJsonExtraInformationError(Exception):
    def __init__(self):
        self.message: str = "The Json doesn't contain the files you want to analyse. This is currently not supported. "\
                            "Please make sure to add the correct path to the json file."

    def __str__(self):
        return f"{self.message}"


class MolecularityRegexError(Exception):
    def __init__(self):
        self.message: str = "The desired molecularity is not implemented. Check your Json for the correct spelling."

    def __str__(self):
        return f"{self.message}"


class ExpectedTransitionsError(Exception):
    def __init__(self):
        self.message: str = "The desired expected transisions are not supported. Currently only one or two are supported."

    def __str__(self):
        return f"{self.message}"