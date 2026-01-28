from datetime import datetime
import os


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def get_kernel(height, width):
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size


def get_width_height(patch_info):
    w_input = int(patch_info.split('x')[-1])
    h_input = int(patch_info.split('x')[0].split('_')[-1])
    return w_input,h_input


def parse_model_name(model_name):
    """Parse model name to extract input size, model type, and scale."""
    basename = os.path.basename(model_name).replace('.pth', '')
    parts = basename.split('_')

    h_input = w_input = None
    model_type = None
    scale = None

    for part in parts:
        if 'x' in part and len(part.split('x')) == 2:
            try:
                h_input, w_input = map(int, part.split('x'))
            except ValueError:
                raise ValueError(f"Invalid resolution format: {part}")
        elif part.replace('.', '', 1).isdigit():
            scale = float(part)
        else:
            model_type = part

    if h_input is None or w_input is None:
        raise ValueError(f"❌ Model name must include resolution like '80x80': {model_name}")

    return h_input, w_input, model_type, scale



def make_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)