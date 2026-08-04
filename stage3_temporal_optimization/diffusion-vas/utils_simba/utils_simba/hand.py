import numpy as np
import smplx

def initialize_mano_model(mano_r_dir: str):
    """
    Initializes the MANO model for the right hand.

    Parameters:
        mano_r_dir (str): Path to the MANO model directory.

    Returns:
        tuple: Faces of the MANO model as a NumPy array.
    """
    
    mano_model = smplx.create(
        model_path=mano_r_dir,
        model_type="mano",
        use_pca=False,
        is_rhand=True
    )
    f3d_r = np.array(mano_model.faces)
    return f3d_r

