ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg'}


def allowed_file(filename: str) -> bool:
    """
    Check if the file extension is allowed.

    Args:
        filename (str): The filename to check.

    Returns:
        bool: True if the file extension is allowed, False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

