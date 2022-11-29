from pathlib import Path


def unravel(lst):
	if isinstance(lst, list):
		result = []
		for elem in lst:
			result.extend(unravel(elem))
		return result
	elif isinstance(lst, int):
		return [lst]
	else:
		return []

def get_files(input_path):
    supported_file_types = ['jsonl.zst']
    
    if isinstance(input_path, str):
        input_path = Path(input_path)
    
    if input_path.is_dir():
        # get all files with supported file types
        files = [list(Path(input_path).glob(f"*{ft}")) for ft in supported_file_types]
        # flatten list
        files = [f for sublist in files for f in sublist]
        assert files, f"No files with supported types found in directory: {input_path}"
    elif input_path.is_file():
        assert any(
            str(input_path).endswith(f_type) for f_type in supported_file_types
        ), f"Input file type must be one of: {supported_file_types}"
        files = [input_path]
    else:
        raise FileNotFoundError(f"No such file or directory: {input_path}")

    return [str(f) for f in files]


if __name__=='__main__':
    print(get_files('test'))
